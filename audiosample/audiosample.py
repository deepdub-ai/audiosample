import io
import struct
import sys
import warnings
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Optional, Union, Generator, Any, Dict, List, Tuple, BinaryIO, cast
from logging import getLogger
from types import GeneratorType

logger = getLogger(__name__)

import numpy as np

try:
    import av
    AV_SUPPORTED = True
except ImportError:
    warnings.warn("AudioSample unable to support ffmpeg bindings, please install PyAv")
    AV_SUPPORTED = False

try:
    import torch
    TORCH_SUPPORTED = True
except ImportError:
    warnings.warn("AudioSample unable to support torch tensors, please install torch")
    TORCH_SUPPORTED = False

RIFF_HEADER_MAGIC = b"RIFF"
RIFF_HEADER_LEN = 8
JUST_WAVE_HEADER_LEN = 4
CHUNK_HEADER_LEN = 8
FORMAT_HEADER_LEN = 40
SAMPLE_TO_FLOAT_DIVISOR = 2 ** 15
SAMPLE_TO_FLOAT_DIVISOR_24BIT = 2 ** 23
SAMPLE_TO_FLOAT_DIVISOR_32BIT = 2 ** 31
RIFF_HEADER_STRUCT = "<4sI"
JUST_WAVE_HEADER_STRUCT = "<4s"
CHUNK_HEADER_STRUCT = "<4sI"
FORMAT_HEADER_CONTENT_STRUCT = "HHIIHH"
FORMAT_HEADER_CONTENT_LEN = 16
FORMAT_HEADER_EX_STRUCT = "24s"
FORMAT_HEADER_EX_LEN = 24
FORMAT_HEADER_FULL_STRUCT = FORMAT_HEADER_CONTENT_STRUCT + FORMAT_HEADER_EX_STRUCT
FORMAT_HEADER_FULL_LEN = FORMAT_HEADER_CONTENT_LEN + FORMAT_HEADER_EX_LEN
DATA_HEADER_STRUCT = CHUNK_HEADER_STRUCT
DATA_HEADER_LEN = CHUNK_HEADER_LEN
# Combine headers, and  remove little endian sign.
WAVE_HEADER_STRUCT = RIFF_HEADER_STRUCT + (JUST_WAVE_HEADER_STRUCT + CHUNK_HEADER_STRUCT + FORMAT_HEADER_CONTENT_STRUCT + FORMAT_HEADER_EX_STRUCT + DATA_HEADER_STRUCT).replace("<","")
WAVE_HEADER_LEN = RIFF_HEADER_LEN + JUST_WAVE_HEADER_LEN + CHUNK_HEADER_LEN + FORMAT_HEADER_FULL_LEN + DATA_HEADER_LEN
WAVE_ACCEPTED_PRECISIONS = [16, 24, 32]
DEFAULT_PRECISION = 16
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1
DEFAULT_WAVE_FORMAT = 1
PRECISION_CODECS: Dict[int, str] = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}
PRECISION_FORMATS: Dict[int, str] = {16: "s16le", 24: "s24le", 32: "s32le"}
WaveHeader = namedtuple("WaveHeader", ["riff_header", "total_length", "wave_header",
                                       "format_header", "format_data_length", "type_of_format", "channels",
                                       "sample_rate", "size_per_second", "channel_sample_width", "precision", "nulldata", "data_header",
                                       "data_length"])
NotWaveHeader = namedtuple("NotWaveHeader", [ "sample_rate", "channels", "streams", "precision"])


class NotRIFF(Exception):
    """
    Exception raised when the file is not a RIFF file.
    """
    pass


class AudioSample:
    """AudioSample is a class that lazy manipulates audio data. It is highly optimize so as to only render what it needs when it needs.
    AudioSample is native to wav/RIFF files, but can also read from a variety of other formats using the ffmpeg bindings provided by PyAv.
    It can be used to read, write, and manipulate audio data in a variety of formats. 
    - It can be used to read audio data from a file, a URL, or a byte stream. It can also be used to write audio data to a file or a byte stream. 
    - It can be used to manipulate audio data in a variety of ways, such as converting between mono and stereo, resampling, and normalizing. 
    - It can also be used to display audio data in a variety of ways, such as as a spectrogram or as a waveform. 
    - It can also be used to play audio data in a variety of ways, such as through a speaker or through headphones. 
    - It can be used to convert audio data between a variety of formats, such as between WAV and MP3. 
    - It can be used to convert audio data between a variety of sample rates, such as between 44.1 kHz and 48 kHz. 
    - It can be used to convert audio data between a variety of bit depths, such as between 16-bit and 24-bit. 
    - It can be used in conjunction with other libraries, such as NumPy, SciPy, and PyTorch.
    - It can be used to save audio data to a file or a byte stream.
    To initialize an AudioSample object, you can pass in a file path, a URL, or a byte stream. e.g.:
    ```
    au = AudioSample("path/to/file.wav")
    ```
    To load the audio from numpy:
    ```
    au = AudioSample(np.array([1,2,3,4,5]))
    ```
    To load the audio from a torch tensor:
    ```
    au = AudioSample(torch.tensor([1,2,3,4,5]))
    ```
    To load the audio from a URL:
    ```
    au = AudioSample("https://example.com/audio.wav")
    ```
    To save the audio to a file:
    ```
    au.write("path/to/file.wav")
    You can also use the `as_wav_data` method to get the audio data as a byte stream:
    ```
    data = au.as_wav_data()
    ```
    You can also save a m4a file:
    ```
    au.write("path/to/file.m4a")
    ```
    You can also save as a numpy array:
    ```
    data = au.as_numpy()
    ```
    You can also save as a torch tensor:
    ```
    data = au.as_torch()
    ```
    
    AudioSample doesn't close files until the object is destroyed. If you want to close the file, you can also use the 'with' statement:
    ```
    with AudioSample("path/to/file.wav") as au:
        au.play()
    ```
    Reading from multiple streams:
    To read multiple streams from a file, you can first initialize an AudioSample object, and get the number of streams in the file using the `streams` attribute. 
    You can then read from each specific stream by using the `stream_idx` parameter:
    ```
    #write each stream of an m4a to a different file in m4a and wav format.
    au = AudioSample("path/to/file.m4a")
    n_streams = au.streams
    for i in range(n_streams):
        au_i = AudioSample("path/to/file.m4a", stream_idx=i)
        au_i.write(f"path/to/file_stream_{i}.m4a", no_encode=True)
        au_i.write(f"path/to/file_stream_{i}.wav")
    ```

    """
    unit_sec: bool = True

    def __init__(self, f: Union[str, bytes, Path, io.BytesIO, io.FileIO, GeneratorType, None]=None, 
                 force_read_format: Optional[str]=None, force_sample_rate: Optional[int]=None, 
                 force_channels: Optional[int]=None, force_precision: Optional[int]=None, 
                 unit_sec: Optional[bool]=None, thread_safe: bool=False, stream_idx: int=0, 
                 force_read_sample_rate: Optional[int]=None, force_read_channels: Optional[int]=None, 
                 force_read_precision: Optional[int]=None, force_bit_rate: Optional[int]=None) -> None:
        """
        Initialize an AudioSample object.
        Parameters
        ----------
        f : str, bytes, Path, io.BytesIO, io.BufferedReader, io.BufferedRandom, io.FileIO, io.RawIOBase, or None
            The audio data to load. If `f` is a string, it is treated as a file path. If `f` is a bytes object, it is treated as audio data. If `f` is a Path object, it is treated as a file path. If `f` is a file-like object, it is treated as audio data. If `f` is None, an empty AudioSample object is created.
        force_read_format : str, optional
            The format to read the audio data in. If `force_read_format` is None, the format is determined automatically. If `force_read_format` is not None, the format is forced to be `force_read_format`.
        force_sample_rate : int, optional
            The sample rate to read the audio data in. If `force_sample_rate` is None, the sample rate is determined automatically. If `force_sample_rate` is not None, the sample rate is forced to be `force_sample_rate`.
        force_channels : int, optional
            The number of channels to read the audio data in. If `force_channels` is None, the number of channels is determined automatically. If `force_channels` is not None, the number of channels is forced to be `force_channels`.
        force_precision : int, optional
            The bit depth to read the audio data in. If `force_precision` is None, the bit depth is determined automatically. If `force_precision` is not None, the bit depth is forced to be `force_precision`.
        unit_sec : bool, optional
            Whether to manipulate in sample units or seconds units. If `unit_sec` is True, the indexes are treats a floating point seconds. If `unit_sec` is False, the indexes are treated as integer samples.
        thread_safe : bool, optional
            Whether to make the AudioSample object thread safe. If `thread_safe` is True, the copy of AudioSample object can be passed safely to other threads. If `thread_safe` is False, the AudioSample object is not thread safe. You can not manipulate
            the same AudioSample object in multiple threads. You can easily create a new AudioSample object using:
            ```
            Thread.start(target=do_something, args=(au[:],))
            ```
        stream_idx : int, optional
            The index of the audio stream to read. If `stream_idx` is None, the first audio stream is read. If `stream_idx` is not None, the audio stream at index `stream_idx` is read.
        """
        self._data: bytes = b''
        self.start: int = 0
        self.stream_idx: int = stream_idx
        self.data_start: int = -1
        self.len: int = 0
        self.unit_sec: bool = unit_sec if unit_sec is not None else self.__class__.unit_sec
        self.thread_safe: bool = thread_safe
        self.wave_header: Optional[WaveHeader] = None
        self.not_wave_header: Optional[NotWaveHeader] = None
        self.input_container: Optional[Any] = None
        self.force_read_format: Optional[str] = None
        self.force_sample_rate: Optional[int] = None
        self.force_channels: Optional[int] = None
        self.force_precision: Optional[int] = None
        self.iterable_input_buffer: Optional[Generator] = None
        self.force_read_sample_rate: Optional[int] = force_read_sample_rate
        self.force_read_channels: Optional[int] = force_read_channels
        self.force_read_precision: Optional[int] = force_read_precision
        self.force_bit_rate: Optional[int] = force_bit_rate
        self.layout_possibilities: Dict[int, str] = { 1: "mono", 2: "stereo", 3: "3.0", 4: "quad", 5: "5.0", 6: "5.1", 7: "6.1", 8: "7.1", 9: "7.1(wide)", 10: "7.1(wide-side)", 11: "7.1(top-front)", 12: "7.1(top-front-wide)", 13: "7.1(top-front-high)", 14: "7.1(top-front-high-wide)", 15: "7.1(top-front-high-wide-side", 16: "7.1(top-front-high-wide-side-rear)"}
        
        self.f: Optional[Union[str, bytes, Path, io.BytesIO, io.FileIO, GeneratorType]] = None

        if f is not None:
            if isinstance(f, np.ndarray):
                if not force_read_sample_rate:
                    raise ValueError("force_read_sample_rate must be provided if numpy array is provided.")

                f = AudioSample.from_numpy(f, rate=force_read_sample_rate)
                self.__setstate__(f.__getstate__())
                if force_sample_rate or force_channels or force_precision:
                    self.force_sample_rate = force_sample_rate
                if force_channels:
                    self.force_channels = force_channels
                if force_precision:
                    self.force_precision = force_precision

            elif isinstance(f, AudioSample):
                self.__setstate__(f.__getstate__())
            else:
                self._open(f, force_read_format=force_read_format, 
                       force_sample_rate=force_sample_rate, force_channels=force_channels, force_precision=force_precision, force_bit_rate=force_bit_rate)
        else:
            #create empty AudioSample
            precision = force_precision if force_precision else DEFAULT_PRECISION
            sample_rate = force_sample_rate if force_sample_rate else DEFAULT_SAMPLE_RATE
            channels = force_channels if force_channels else DEFAULT_CHANNELS
            channel_sample_width = channels * (precision // 8)
            size_per_second = sample_rate * channel_sample_width
            self.wave_header = WaveHeader(riff_header=b"RIFF", total_length=WAVE_HEADER_LEN - RIFF_HEADER_LEN + DATA_HEADER_LEN, wave_header=b"WAVE", format_header=b"fmt ", format_data_length=FORMAT_HEADER_FULL_LEN,
                                          type_of_format=DEFAULT_WAVE_FORMAT, 
                                          channels=channels, sample_rate=sample_rate, size_per_second=size_per_second, channel_sample_width=channel_sample_width,  
                                          precision=precision, nulldata=b"\x00"*24, data_header=b"data", data_length=0)
            self.data = b''
            self.data_start = 0
    
    @classmethod
    def set_default_unit_samples(cls) -> None:
        """
        Set the default unit_sec usage.
        """
        cls.unit_sec = False
    
    @classmethod
    def set_default_unit_sec(cls) -> None:
        """
        Set the default unit_sec usage.
        """
        cls.unit_sec = True

    def unit_samples_(self, unit_samples: bool=True) -> 'AudioSample':
        """
        Set the unit_sec usage in place.
        """
        self.unit_sec = not unit_samples
        return self
    # Inplace set unit_sec usage. returns self.
    def unit_sec_(self, unit_sec: bool=True) -> 'AudioSample':
        """
        Set the unit_sec usage in place.
        """
        self.unit_sec = unit_sec
        return self

    def __enter__(self) -> 'AudioSample':
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.cleanup()

    def _open(self, f: Union[str, bytes, Path, io.BytesIO, io.FileIO, GeneratorType], 
              force_read_format: Optional[str]=None, force_sample_rate: Optional[int]=None, 
              force_channels: Optional[int]=None, force_precision: Optional[int]=None, force_bit_rate: Optional[int]=None) -> None:
        self.f = f
        self.force_read_format = force_read_format
        self.force_sample_rate = force_sample_rate
        self.force_channels = force_channels
        self.force_precision = force_precision
        self.force_bit_rate = force_bit_rate

        if isinstance(f, bytes):
            f = self.f = io.BytesIO(f)

        if isinstance(f, Path):
            f = self.f = str(self.f)

        if isinstance(f, str):
            if f.startswith("https://") or f.startswith("http://"):
                self._open_with_av()
                return
            fname = self.f
            f = self.f = open(fname, "rb")

        if isinstance(f, io.BytesIO):
            f.seek(0, 0)
        if isinstance(f, GeneratorType):

            if self.thread_safe:
                raise ValueError("Cannot create thread safe AudioSample from iterable.")
            self.iterable_input_buffer = f
            try:
                first_chunk = next(self.iterable_input_buffer)
            except StopIteration:
                raise ValueError("Empty generator provided.")
            if isinstance(first_chunk, bytes):
                if len(first_chunk) == 0:
                    raise ValueError("Empty bytes provided.")
                f = self.f = io.BytesIO(first_chunk)
            else:
                if isinstance(first_chunk, np.ndarray):
                    if not self.force_read_sample_rate:
                        raise ValueError("force_read_sample_rate must be provided if numpy array is provided.")
                    first_chunk = AudioSample.from_numpy(first_chunk, rate=self.force_read_sample_rate)
                if isinstance(first_chunk, AudioSample):
                    first_chunk.read()
                    f = self.f = io.BytesIO(first_chunk._data)
                    self.force_channels = self.force_channels or first_chunk.channels
                    self.force_sample_rate = self.force_sample_rate or first_chunk.sample_rate
                    self.force_precision = self.force_precision or first_chunk.precision
                else:
                    raise ValueError(f"Unsupported generator type {type(first_chunk)=}")

            self.force_read_format = PRECISION_FORMATS[DEFAULT_PRECISION]
            if self.force_read_precision:
                if self.force_read_precision not in WAVE_ACCEPTED_PRECISIONS:
                    raise ValueError(f"Unsupported precision {self.force_read_precision=}")
                self.force_read_format = PRECISION_FORMATS[self.force_read_precision]
        
        if not (getattr(f, 'seek', None) and getattr(f, 'read', None)):
            raise ValueError(f"f must support file like behavior: {f=}")

        if not isinstance(f, io.BytesIO) and not (getattr(f, 'mode', None) and "b" in f.mode):
            raise ValueError(f"f must be opened in binary mode: {f=}")

        try:
            if force_sample_rate or force_channels:
                raise NotRIFF()

            self.wave_header = self.read_file_header(f)
            self.len = self.wave_header.data_length // self.wave_header.channel_sample_width
            return
        except NotRIFF:
            if not AV_SUPPORTED:
                raise

        self._open_with_av()
        
    def _create_input_container(self) -> None:
        try:
            if self.force_read_format or self.force_read_channels or self.force_read_sample_rate or self.force_read_precision:
                kwargs: Dict[str, Dict[str, str]] = defaultdict(dict)
                if self.force_read_sample_rate:
                    kwargs['options']['sample_rate'] = str(self.force_read_sample_rate)
                if self.force_read_channels:
                    kwargs['options']['channels'] = self.layout_possibilities[self.force_read_channels]
                self.input_container = av.open(self.f, format=self.force_read_format, mode='r', metadata_errors='ignore', **kwargs)
            else:
                self.input_container = av.open(self.f, metadata_errors='ignore')
            self.input_container.flags |= av.container.core.Flags.FAST_SEEK
            
        except av.error.InvalidDataError:
            raise ValueError(f"Corrupt data or header")
        # | av.container.core.Flags.NOBUFFER | av.container.core.Flags.NONBLOCK


    def _open_with_av(self) -> None:
        self._create_input_container()
        self.n_streams = len(self.input_container.streams.audio)
        if self.stream_idx >= self.n_streams:
            raise ValueError(f"Requested non-existent stream, choose from {range(self.n_streams)}")
        input_stream = self.input_container.streams.audio[self.stream_idx]
        out_sample_rate = self.force_sample_rate if self.force_sample_rate else input_stream.codec_context.sample_rate
        out_channels = self.force_channels if self.force_channels else input_stream.codec_context.channels
        out_precision = self.force_precision if self.force_precision else DEFAULT_PRECISION
        assert out_precision in WAVE_ACCEPTED_PRECISIONS, "Unsupported precision"
        self.not_wave_header = NotWaveHeader(out_sample_rate, out_channels, self.n_streams, out_precision)
        self.len = int(input_stream.duration*input_stream.time_base*out_sample_rate)
        start_time = 0 if input_stream.start_time is None else input_stream.start_time
        self.start = int((start_time*input_stream.time_base)*out_sample_rate)

    @classmethod
    def _get_codec_from_format_name(cls, format_name: Optional[str]) -> Optional[str]:
        if format_name is None:
            return None
        if "mp3" in format_name:
            return "mp3"
        elif "wav" in format_name or "s16le" in format_name:
            return "pcm_s16le"
        elif "f32le" in format_name:
            return "pcm_f32le"
        elif "s24le" in format_name:
            return "pcm_s24le"
        elif "mulaw" in format_name:
            return "pcm_mulaw"
        elif "ogg" in format_name or "opus" in format_name:
            return "libvorbis"
        elif "mp4" in format_name or "ipod" in format_name or "m4a" in format_name or "mov" in format_name or "m4b" in format_name or "ts" in format_name or "aac" in format_name or "adts" in format_name:
            return "aac"
        return format_name

    def _read_with_av(self, out_file: Optional[Union[str, Path, io.BytesIO]]=None, 
                     force_out_format: Optional[str]=None, no_encode: bool=False) -> Generator[bytes, None, None]:
        if self.len == 0:
            raise ValueError("No data in audiosample provided.")
        def get_format_from_format_name(format_name: str) -> str:
            if "mp4" in format_name:
                return "mp4"
            if "mov" in format_name:
                return "mov"
            if "ts" in format_name:
                return "mpegts"
            if "mulaw" in format_name:
                return "wav"
            return format_name
        def get_format_name_from_user_given_name(format_name: str) -> str:
            if "m4a" in format_name:
                return "ipod"
            if "ts" in format_name:
                return "mpegts"
            if "aac" in format_name:
                return "adts"
            return format_name
                    
        real_out_file: Optional[BinaryIO] = None
        out_file = io.BytesIO() if not out_file else out_file
        seek_to_0_once = False
        if not self.f:
            self.f = io.BytesIO(self.as_wav_data())
            self.f.seek(0,0)
            self._create_input_container()
            seek_to_0_once = True
        elif not self.input_container:
            self.f.seek(0,0) if hasattr(self.f, 'seek') else None
            self._create_input_container()
            seek_to_0_once = True

        if self.__class__._get_codec_from_format_name(force_out_format) != self.__class__._get_codec_from_format_name(self.format):
            no_encode = False
        if self.force_channels or self.force_sample_rate:
            no_encode = False
        if self.iterable_input_buffer:
            real_out_file = out_file
            out_file = io.BytesIO()
            last_output_read_pos = 0

        out_format = get_format_name_from_user_given_name(force_out_format) if force_out_format else get_format_from_format_name(self.input_container.format.name)
        output_container = av.open(out_file, format=out_format, mode='w', metadata_errors='ignore')
        input_stream = self.input_container.streams.audio[self.stream_idx]
        start_sec = self.start / self.sample_rate
        stop_sec = (self.start + self.len) / self.sample_rate

        start_time = 0 if input_stream.start_time is None else input_stream.start_time*input_stream.time_base
        seek_to = int(max((start_sec-0.5), start_time)/input_stream.time_base)
        if seek_to == start_time and start_time == 0:
            if not seek_to_0_once:
                self.f.seek(0,0) if hasattr(self.f, 'seek') else None
                self._create_input_container()
        else:
            self.input_container.seek(seek_to, 
                                stream=input_stream)
        if not no_encode:
            output_stream = output_container.add_stream(self.__class__._get_codec_from_format_name(out_format), 
                                    rate=self.force_sample_rate or self.sample_rate, layout=self.layout_possibilities[self.channels])
            
            if self.force_bit_rate:
                output_stream.bit_rate = self.force_bit_rate
            codec = output_stream.codec_context
            frame_ts_start = None
            for packet in self.input_container.demux(input_stream):
                try:
                    if packet.dts is None:
                        continue
                    if not self.iterable_input_buffer:
                        if ((packet.dts+packet.duration)*input_stream.time_base) < start_sec:
                            for frame in packet.decode():
                                pass
                            continue                    
                        if packet.dts and (packet.dts*input_stream.time_base) >= stop_sec:
                            break
                    
                    for frame in packet.decode():
                        if frame.dts is None or (frame.dts*input_stream.time_base+frame.samples/frame.sample_rate) <= start_sec:
                            del frame
                            continue
                        
                        if not self.iterable_input_buffer:
                            if (frame.dts*input_stream.time_base) >= stop_sec:
                                break

                        if frame_ts_start is None:
                            frame_ts_start = frame.dts

                        frame.pts = (frame.dts - frame_ts_start)

                        try:
                            encoded_packets = codec.encode(frame)
                        except ValueError as ve:
                            logger.warning(f"Received ValueError {ve=} restarting codec.")
                            output_stream.codec_context.close()
                            _codec = av.CodecContext.create(self.__class__._get_codec_from_format_name(out_format), mode="w")
                            _codec.channels = codec.channels
                            _codec.rate = codec.rate
                            _codec.format = codec.format
                            del codec
                            codec = _codec
                            encoded_packets = codec.encode(frame)
                        if encoded_packets:
                            output_container.mux(encoded_packets)                            
                        del encoded_packets
                        del frame
                except av.error.InvalidDataError:                                                                                                                                                                                           
                    logger.warning(f"Invalid packet found while processing {self.f}")
                    pass
                except ValueError:
                    logger.warning(f"Invalid packet found while processing {self.f}")
                    pass
                if self.iterable_input_buffer:
                    #save the original position of the av reader
                    av_reader_pos = self.f.tell()
                    MIN_PACKETS = 2
                    while (packet.pos + packet.size*MIN_PACKETS) > len(self.f.getvalue()):
                        try:
                            chunk = next(self.iterable_input_buffer)
                            if isinstance(chunk, bytes):
                                self.f.write(chunk)
                            else:
                                if isinstance(chunk, np.ndarray):
                                    chunk = AudioSample.from_numpy(chunk, rate=self.force_read_sample_rate)
                                    if chunk.channels != self.channels:
                                        raise ValueError(f"Channels mismatch {chunk.channels=} {self.channels=}")
                                if isinstance(chunk, AudioSample):
                                    chunk.read()
                                    self.f.write(chunk._data)
                                else:
                                    raise ValueError(f"Unsupported iterator type {type(chunk)=}")
                        except StopIteration:
                            break
                    #go back to the original position
                    self.f.seek(av_reader_pos, 0)
                    chunk_out = out_file.getvalue()[last_output_read_pos:]
                    if real_out_file:
                        real_out_file.write(chunk_out)
                    last_output_read_pos = len(out_file.getvalue())
                    if (len(chunk_out) > 0):
                        yield chunk_out
            del codec
            del packet
            output_container.mux(output_stream.encode(None))
        else:
            output_stream = output_container.add_stream(template=input_stream)            
            packet_ts_start = None
            for packet in self.input_container.demux(input_stream):
                if packet.dts is None:
                    continue
                if not self.iterable_input_buffer:
                    if ((packet.dts+packet.duration)*input_stream.time_base) < start_sec:
                        continue
                    if (packet.dts*input_stream.time_base) >= stop_sec:
                        break
                if packet_ts_start is None:
                    packet_ts_start = packet.dts
                if packet.stream_index != 0:
                    pkt = av.packet.Packet(bytes(packet))
                else:
                    pkt = packet
                pkt.dts = pkt.pts = (packet.dts - packet_ts_start)
                output_container.mux(pkt)
                del packet


        output_container.close()
        wav_buffer = b''
        if getattr(out_file, 'getvalue', None):
            wav_buffer = out_file.getvalue()
        if out_file.tell() == 0:
            raise ValueError("No data in audiosample provided.")

        del output_container
        del out_file
        if self.iterable_input_buffer:
            if real_out_file:
                real_out_file.write(wav_buffer[last_output_read_pos:])
            yield wav_buffer[last_output_read_pos:]
        else:
            yield wav_buffer

    def _read_wav_with_av(self, out_file=None):
        if self.len == 0:
            raise ValueError("No data in audiosample provided.")
        if not out_file:
            out_file = io.BytesIO()
        seek_to_0_once = False    
        if not self.input_container:
            self.f.seek(0,0) if hasattr(self.f, 'seek') else None
            self._create_input_container()
            seek_to_0_once = True

        output_container = av.open(out_file, format='wav', mode='w', metadata_errors="ignore")
        input_stream = self.input_container.streams.audio[self.stream_idx]
        start_sec = self.start / self.sample_rate
        stop_sec = (self.start + self.len) / self.sample_rate

        #first frame decoded is always 0.
        start_time = 0 if input_stream.start_time is None else input_stream.start_time*input_stream.time_base
        seek_to = int(max((start_sec-0.5), start_time)/input_stream.time_base)
        if seek_to == start_time and start_time == 0:
            if not seek_to_0_once:
                self.f.seek(0,0) if hasattr(self.f, 'seek') else None
                self._create_input_container()
        else:
            self.input_container.seek(seek_to, 
                                stream=input_stream)

        output_stream = output_container.add_stream(PRECISION_CODECS[self.not_wave_header.precision], rate=self.force_sample_rate or self.not_wave_header.sample_rate, layout=self.layout_possibilities[self.not_wave_header.channels])
        # output_stream.channels = self.not_wave_header.channels
        codec = output_stream.codec_context
        actual_start_due_to_frame = -1
        for packet in self.input_container.demux(input_stream):
            try:
                if packet.dts is not None and (packet.dts*input_stream.time_base) >= stop_sec:
                    break
                if packet.dts is not None and ((packet.dts+packet.duration)*input_stream.time_base) < start_sec:
                    for frame in packet.decode():
                        pass
                    continue
                for frame in packet.decode():
                    if actual_start_due_to_frame == -1:
                        actual_start_due_to_frame = frame.dts*input_stream.time_base
                    try:
                        encoded_frame = codec.encode(frame)
                    except ValueError as ve:
                        logger.warning(f"Received ValueError {ve=} restarting codec.")
                        output_stream.codec_context.close()
                        _codec = av.CodecContext.create("pcm_s16le", mode="w")
                        _codec.channels = codec.channels
                        _codec.rate = codec.rate
                        _codec.format = codec.format
                        del codec
                        codec = _codec
                        encoded_frame = codec.encode(frame)
                    output_container.mux(encoded_frame)
                    del encoded_frame
                    del frame
            except av.error.InvalidDataError:
                logger.warning(f"Invalid packet found while processing {self.f}")
                pass
            except ValueError:
                logger.warning(f"Invalid packet found while processing {self.f}")
                pass

            del packet
        
        #flush packets.
        output_container.mux(output_stream.encode(None))
        output_container.close()
        wav_buffer = None
        if getattr(out_file, 'getvalue', None):
            wav_buffer = out_file.getvalue()
        if out_file.tell() == 0:
            raise ValueError("No data in audiosample provided.")
        out_file.close()
        del out_file
        del output_container
        return wav_buffer, int(actual_start_due_to_frame*self.not_wave_header.sample_rate) if actual_start_due_to_frame >= 0 else 0

    @property
    def streams(self):
        """
        The number of audio streams in the file.
        """
        if self.wave_header:
            return 1
        if self.not_wave_header:
            return self.not_wave_header.streams
        raise ValueError("Unavailable data.")

    @property
    def sample_rate(self) -> int:
        """
        The sample rate of the audio data.
        """
        if self.wave_header:
            return self.wave_header.sample_rate
        return self.not_wave_header.sample_rate

    @property
    def precision(self) -> int:
        """
        The bit depth of the audio data.
        """
        if self.wave_header:
            return self.wave_header.precision
        return self.not_wave_header.precision

    @property
    def start_time(self):
        """
        The start time of the audio data in seconds.
        """
        return self.start/self.sample_rate

    @property
    def sample_width(self):
        """
        The sample width of the audio data in bytes.
        """
        return self.precision // 8
    @property
    def channel_sample_width(self):
        """
        The sample width of the audio data in bytes.
        """
        if self.wave_header:
            return self.wave_header.channel_sample_width
        return self.not_wave_header.channels * (self.not_wave_header.precision // 8)
    @property
    def channels(self) -> int:
        """
        The number of channels in the audio data.
        """
        if self.wave_header:
            return self.wave_header.channels
        return self.not_wave_header.channels

    @property
    def format(self):
        """
        The format of the audio data.
        """
        if self.wave_header:
            return 'wav'
        elif self.not_wave_header:
            if 'mp4' in self.input_container.format.name:
                return 'mp4'
            return self.input_container.format.name
        raise ValueError("File not open")

    @property
    def duration(self) -> float:
        """
        The duration of the audio data in seconds.
        """
        return self.len / self.sample_rate

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise ValueError(f"{self.__class__.__name__} only accepts slices e.g au[0:1]")
        if self.unit_sec:
            index_start = index.start if not index.start is None else 0
            index_stop = index.stop if not index.stop is None else self.duration
            # Turn into sample units.
            if self.wave_header:
                index_start = int(index_start * self.wave_header.sample_rate)
                index_stop = int(index_stop * self.wave_header.sample_rate)
            elif self.not_wave_header:
                index_start = int(index_start * self.not_wave_header.sample_rate)
                index_stop = int(index_stop * self.not_wave_header.sample_rate)
            index = slice(index_start, index_stop, None)

        index_start, index_stop, skip = index.indices(self.len)
        assert index_start <= index_stop
        real_start = self.start + index_start
        real_stop = self.start + index_stop

        new = self.__class__()
        new.unit_sec = self.unit_sec

        if self._data:
            new._data = self._data[real_start * self.wave_header.channel_sample_width:real_stop * self.wave_header.channel_sample_width]
            new.start = 0
            new.f = None
            new.input_container = None
        else:
            new.f = self.f
            new.input_container = self.input_container
            new.start = real_start

        new.len = index_stop - index_start
        new.wave_header = WaveHeader(*(self.wave_header)) if self.wave_header else None
        new.not_wave_header = NotWaveHeader(*(self.not_wave_header)) if self.not_wave_header else None
        new.data_start = self.data_start
        new.force_read_format = self.force_read_format
        new.force_channels = self.force_channels
        new.force_precision = self.force_precision
        new.force_sample_rate = self.force_sample_rate
        new.force_read_format = self.force_read_format
        new.force_read_sample_rate = self.force_read_sample_rate
        new.force_read_channels = self.force_read_channels
        new.force_read_precision = self.force_read_precision
        new.thread_safe = self.thread_safe
        new.stream_idx = self.stream_idx

        if new.thread_safe and new.f:
            if isinstance(self.f, str) and (self.f.startswith("https://") or self.f.startswith("http://")):
                pass
            elif getattr(new.f, 'name', None):
                new.f = open(new.f.name, 'rb')
            elif getattr(new.f, 'getvalue', None):
                new.f = io.BytesIO(new.f.getvalue())
            elif isinstance(new.f, str):
                new.f = open(new.f.name, 'rb')
            else:
                raise ValueError(f"Unsupported file type for thread_safe cloning. {new.f=}")
                
            if new.input_container:
                new.input_container = None
                if isinstance(new.f, str) and not (new.f.startswith("https://") or new.f.startswith("http://")):
                    new.f = open(new.f, 'rb')
                new._open_with_av()
                new.start = real_start
                new.len = index_stop - index_start
        return new

    def clone(self):
        """
        Clone the AudioSample object.
        Returns
        -------
        AudioSample
            A copy of the AudioSample object.
        """
        return self[:]
    
    def __getstate__(self):
        state = self.__dict__.copy()
        f = state['f']
        if f and hasattr(f, 'name'):
            state['f'] = f.name
        elif f and hasattr(f, 'getvalue'):
            state['f'] = f.getvalue()
        state['input_container'] = None
        return state
    def __setstate__(self, state):
        f = state['f']
        self.__dict__.update(state)
        if f is not None:
            self._open(f, force_read_format=self.force_read_format,
                       force_sample_rate=self.force_sample_rate, force_channels=self.force_channels, force_precision=self.force_precision)

    def __len__(self) -> int:
        """
        The length of the audio data.
        The length is measured in samples.
        """
        return self.len

    def __add__(self, other):
        """
            Concatenate two AudioSamples together. The two AudioSamples must have the same sample rate and number of channels.
        """
        if not isinstance(other, AudioSample):
            raise ValueError("Can only add AudioSample to another AudioSample")
        if self.sample_rate != other.sample_rate:
            raise ValueError("Cannot add AudioSamples with different sample rates")
        if self.channels != other.channels:
            raise ValueError("Cannot add AudioSamples with different number of channels")

        return AudioSample.from_headerless_data(self.read() + other.read(), self.sample_rate, self.precision, self.channels, self.unit_sec)

    def read(self) -> bytes:
        if not self.wave_header:
            wav_data, read_start = self._read_wav_with_av()
            skip_start = self.start - read_start if (self.start - read_start) > 0 else 0
            max_len = self.len
            #_read_wav_data modifies self.len, we want to restore it after...
            self._read_wav_data(wav_data)
            self._data = self._data[skip_start*self.wave_header.channel_sample_width:][:max_len*self.wave_header.channel_sample_width]
            actual_len = len(self._data)//self.wave_header.channel_sample_width
            if max_len > actual_len:
                logger.debug(f"Received ({actual_len=}) data length is not exactly as expected {max_len=} adapting len to reflect!")
                self.len = actual_len
            elif max_len < actual_len:
                 self._data = self._data[:self.len*self.wave_header.channel_sample_width]
            else:
                self.len = max_len
            return self._data
        if len(self._data) != (self.wave_header.channel_sample_width * len(self)):
            self._data = self.read_file_from_to(self.start, self.len)
        return self._data

    def read_file_header(self, f):
        f.seek(0, 0)
        riff_header = f.read(RIFF_HEADER_LEN)

        if len(riff_header) != RIFF_HEADER_LEN:
            raise Exception("File should be minimum 8 bytes")

        if riff_header[0:4] != RIFF_HEADER_MAGIC:
            f.seek(0, 0)
            raise NotRIFF()

        riff_header_struct = list(struct.unpack(RIFF_HEADER_STRUCT, riff_header))
        wave_header_text = f.read(JUST_WAVE_HEADER_LEN)
        wave_header_text = list(struct.unpack(JUST_WAVE_HEADER_STRUCT, wave_header_text))
        # Look for format header.
        st_tmp = ['', 0]

        while st_tmp[0] != b'fmt ':
            chunk_header = f.read(CHUNK_HEADER_LEN)
            st_tmp = list(struct.unpack(CHUNK_HEADER_STRUCT, chunk_header))
            chunk = f.read(st_tmp[-1])

        fmt_header = st_tmp + list(struct.unpack(FORMAT_HEADER_CONTENT_STRUCT, chunk[:16]))
        if fmt_header[5] != 1:
            # only PCM supported.
            # extensible format is supported through PyAV.
            f.seek(0,0)
            raise NotRIFF()
        if fmt_header[7] not in WAVE_ACCEPTED_PRECISIONS:
            f.seek(0,0)
            raise NotRIFF()
        # look for data header
        fmt_header += [b"\x00"*24]
        chunk_header = f.read(CHUNK_HEADER_LEN)
        st_tmp = list(struct.unpack(CHUNK_HEADER_STRUCT, chunk_header))

        while st_tmp[0] != b'data':
            chunk = f.read(st_tmp[1])
            chunk_header = f.read(CHUNK_HEADER_LEN)
            st_tmp = list(struct.unpack("<4sI", chunk_header))

        riff_header_struct[1] = WAVE_HEADER_LEN - RIFF_HEADER_LEN + st_tmp[1]
        st = riff_header_struct + wave_header_text + fmt_header + st_tmp
        self.data_start = f.seek(0, 1)
        f.seek(0,0) #reset seek.

        return WaveHeader(*st)

    def read_data_header(self, d):
        index = 0
        riff_header = d[index:index+RIFF_HEADER_LEN]; index += RIFF_HEADER_LEN
        riff_header_struct = list(struct.unpack(RIFF_HEADER_STRUCT, riff_header))
        wave_header_text = d[index:index+JUST_WAVE_HEADER_LEN]; index += JUST_WAVE_HEADER_LEN
        wave_header_text = list(struct.unpack(JUST_WAVE_HEADER_STRUCT, wave_header_text))
        # look for format header.
        st_tmp = ['', 0]

        while st_tmp[0] != b'fmt ':
            chunk_header = d[index:index+CHUNK_HEADER_LEN]; index += CHUNK_HEADER_LEN
            st_tmp = list(struct.unpack(CHUNK_HEADER_STRUCT, chunk_header))
            chunk = d[index:index+st_tmp[-1]]; index += st_tmp[-1]

        fmt_header = st_tmp + list(struct.unpack(FORMAT_HEADER_CONTENT_STRUCT, chunk[:16]))
        # look for data header
        fmt_header += [b"\x00"*24] #nulldata
        chunk_header = d[index:index+CHUNK_HEADER_LEN]; index += CHUNK_HEADER_LEN
        st_tmp = list(struct.unpack(CHUNK_HEADER_STRUCT, chunk_header))

        while st_tmp[0] != b'data':
            chunk = d[index:index+st_tmp[1]]; index += st_tmp[1]
            chunk_header = d[index:index+CHUNK_HEADER_LEN]; index += CHUNK_HEADER_LEN
            st_tmp = list(struct.unpack("<4sI", chunk_header))

        riff_header_struct[1] = WAVE_HEADER_LEN - RIFF_HEADER_LEN + st_tmp[1]
        st = riff_header_struct + wave_header_text + fmt_header + st_tmp
        self.data_start = index

        return WaveHeader(*st)

    def read_file_from_to(self, pos, length):
        #assert irrelevant since some files, have bad headers.
        #assert self.data_start >= WAVE_HEADER_LEN
        wh = self.wave_header
        self.f.seek(self.data_start + wh.channel_sample_width * pos, 0)
        data = self.f.read(wh.channel_sample_width * length)
        return data

    def as_numpy(self, mono_1d: bool = True):
        """
        Returns the audio data as a numpy array.
        Parameters
        ----------
        mono_1d : bool, optional
            Whether to return the audio data as a 1D array. If `mono_1d` is True and the audio is 1 channel then data is returned as a 1D array. If `mono_1d` is False, the audio data is returned as a 2D array with dims [ c t ].
        """
        if len(self._data) == 0:
            self.read()
        wh = self.wave_header
        data = self._data
        if wh.channel_sample_width // wh.channels not in [2,3,4]:
            raise ValueError(f"Unsupported bitrate/channel_sample_width {self.f if self.f else ''}")
        if self.wave_header.precision == 32:
            out = np.frombuffer(data, dtype=np.int32)
            out = out.astype('float32') / SAMPLE_TO_FLOAT_DIVISOR_32BIT
        elif self.wave_header.precision == 24:
            out = np.frombuffer(data, dtype=np.int8).reshape(-1, 3)
            out = ((out[:,2].astype('int32') << 16) + (out[:, 1  ].astype('uint8').astype('uint32') << 8) + out[:, 0].astype('uint8').astype('uint32')).astype('float32')
            out /= SAMPLE_TO_FLOAT_DIVISOR_24BIT
        elif self.wave_header.precision == 16:
            out = np.frombuffer(data, dtype='i2').astype('float32')
            out /= SAMPLE_TO_FLOAT_DIVISOR
        else:
            raise ValueError(f"Unsupported precision {self.wave_header.precision}")
        if not mono_1d or wh.channels > 1:
            out = out.reshape((self.len, wh.channels)).transpose()
        return out

    def as_tensor(self, mono_1d: bool = True):
        """
        Returns the audio data as a torch tensor.
        Parameters
        ----------
        mono_1d : bool, optional
            Whether to return the audio data as a 1D tensor. If `mono_1d` is True and the audio is 1 channel then data is returned as a 1D tensor. If `mono_1d` is False, the audio data is returned as a 2D tensor with dims [ c t ].
        """
        if not TORCH_SUPPORTED:
            raise Exception("as_tensor unsupported: pip install torch")

        nout = self.as_numpy(mono_1d=mono_1d)
        out = torch.tensor(nout)
        del nout
        return out

    @classmethod
    def from_numpy(cls, numpy_data: np.ndarray, rate=None, precision=DEFAULT_PRECISION, unit_sec=None):
        """
        Create an AudioSample object from a numpy array.
        Parameters
        ----------
        numpy_data : np.ndarray
            The audio data as a numpy array.
            Number of channels is derived from the shape of the numpy array. 
            - If the numpy array is 1D, the audio data is assumed to be mono. 
            - If the numpy array is 2D, the audio data is assumed to be with the first dimension as the number of channels.
        rate : int, optional
            The sample rate of the audio data. If `rate` is None, the sample rate is set to DEFAULT_SAMPLE_RATE (48000).
        precision : int, optional
            The bit depth of the audio data. If `precision` is None, the bit depth is set to DEFAULT_PRECISION (16).
        unit_sec : bool, optional
            Whether to manipulate in sample units or seconds units. If `unit_sec` is True, the indexes are treats a floating point seconds. If `unit_sec` is False, the indexes are treated as integer samples.
        """
        if not rate:
            warnings.warn("Warning! sample_rate (rate) not specified, defaulting to 16000") if not rate else None
            rate = DEFAULT_SAMPLE_RATE
        assert precision in WAVE_ACCEPTED_PRECISIONS, "Unsupported precision"
        channels = 1 if len(numpy_data.shape) == 1 else numpy_data.shape[0]
        new = cls()
        new.unit_sec = unit_sec if unit_sec is not None else cls.unit_sec
        new.thread_safe = True #no file involved.

        channel_sample_width = channels * (precision // 8)
        new.wave_header = WaveHeader(
            b"RIFF",
            WAVE_HEADER_LEN - RIFF_HEADER_LEN + numpy_data.shape[-1] * channel_sample_width,
            b"WAVE",
            b"fmt ",
            FORMAT_HEADER_FULL_LEN,
            1,  # PCM
            channels,
            rate,
            rate * channel_sample_width,
            channel_sample_width,
            precision,
            b"\x00"*24,
            b"data",
            numpy_data.shape[-1] * channel_sample_width
        )
        new.len = numpy_data.shape[-1]
        if channels > 1:
            numpy_data = numpy_data.transpose()

        if precision == 16:
            new._data = (numpy_data.clip(min=-1, max=(1-1/SAMPLE_TO_FLOAT_DIVISOR)) * SAMPLE_TO_FLOAT_DIVISOR).astype('int16').tobytes()
        elif precision == 24:
            numpy_data = (numpy_data.clip(min=-1, max=(1-1/SAMPLE_TO_FLOAT_DIVISOR_24BIT)) * SAMPLE_TO_FLOAT_DIVISOR_24BIT).astype('int32')
            numpy_data = np.stack([numpy_data & 0xff, (numpy_data >> 8) & 0xff, (numpy_data >> 16)], axis=-1)
            numpy_data = numpy_data.astype('uint8')
            new._data = numpy_data.tobytes()
        elif precision == 32:
            new._data = (numpy_data.clip(min=-1, max=(1-1/SAMPLE_TO_FLOAT_DIVISOR_32BIT)) * SAMPLE_TO_FLOAT_DIVISOR_32BIT).astype('int32').tobytes()

        new.data_start = 0
        new.f = None
        return new

    @classmethod
    def from_tensor(cls, tensor_data, rate, precision=DEFAULT_PRECISION):
        """
        Create an AudioSample object from a torch tensor.
        Parameters
        ----------
        tensor_data : torch.Tensor
            The audio data as a torch tensor.
        rate : int
            The sample rate of the audio data.
        precision : int, optional
            The bit depth of the audio data. If `precision` is None, the bit depth is set to DEFAULT_PRECISION (16).
        """
        if not TORCH_SUPPORTED:
            raise Exception("as_tensor unsupported: pip install torch")
        return cls.from_numpy(tensor_data.numpy(), rate=rate, precision=precision)

    def as_wav_data(self) -> bytes:
        """
        Returns the audio data as a wav file in a bytes object.
        """
        if not self.wave_header:
            self.read()
        wh = WaveHeader(*(self.wave_header))
        st = list(wh)
        st[-1] = self.len * self.wave_header.channel_sample_width
        st[1] = WAVE_HEADER_LEN - RIFF_HEADER_LEN + wh.data_length
        st[4] = FORMAT_HEADER_FULL_LEN
        if not self._data:
            self.read()
        return (struct.pack(WAVE_HEADER_STRUCT, *st) + self._data)

    def as_data_stream(self, out_file=None, no_encode=False, force_out_format=None) -> GeneratorType:
        """
        Returns a generator that yields the audio data in chunks.
        Parameters
        ----------
        out_file : Union[Path, str, io.BytesIO], optional
            The file to write the audio data to. If `out_file` is None, the audio data is returned as a bytes object.
        no_encode : bool, optional
            Whether to encode the audio data. If `no_encode` is True, the audio data is returned as is. If `no_encode` is False, the audio data is encoded or re-encoded.
        force_out_format : str, optional
            The format to encode the audio data to. If `force_out_format` is None, the audio data is returned as is. If `force_out_format` is not None, the audio data is encoded to the specified format.
        Example usage:
        ```
        #get the first second of m4a audio data without re-encoding it.
        audio_sample = AudioSample("path/to/audio.m4a", unit_sec=True)
        audio_1s_data = audio_sample[0:1].as_data(no_encode=True)
        ```
        """
        if (force_out_format and not force_out_format == 'wav') or (self.not_wave_header and no_encode) or self.format != 'wav':
            if self.iterable_input_buffer:
                if force_out_format in ['m4a', 'mp4', 'wav', 'mov']:
                    raise ValueError("Cannot encode stream input to non-streamable output format")
            yield from self._read_with_av(out_file=out_file, force_out_format=force_out_format, no_encode=no_encode)
        else:
            out_buf = self.as_wav_data()
            if out_file:
                out_file.write(out_buf)
            return (yield out_buf)
    
    def as_data(self, out_file: Optional[Union[Path, str, io.BytesIO, io.BufferedWriter]] = None, no_encode: bool = False, force_out_format: Optional[str] = None) -> bytes:
        """
        Returns the audio data as a bytes object or writes it to a file.
        Parameters
        ----------
        out_file : Union[Path, str, io.BytesIO], optional
            The file to write the audio data to. If `out_file` is None, the audio data is returned as a bytes object.
        no_encode : bool, optional
            Whether to encode the audio data. If `no_encode` is True, the audio data is returned as is. If `no_encode` is False, the audio data is encoded or re-encoded.
        force_out_format : str, optional
            The format to encode the audio data to. If `force_out_format` is None, the audio data is returned as is. If `force_out_format` is not None, the audio data is encoded to the specified format.
        Returns
        -------
        Union[bytes, GeneratorType]
            The audio data as a bytes object or a generator that yields chunks of audio data.
        """
        #collect all bytes from as_data
        collect = b''
        gen = self.as_data_stream(out_file=out_file, no_encode=no_encode, force_out_format=force_out_format)
        for chunk in gen:
            collect += chunk
        return collect

    def _read_wav_data(self, data):
        self.wave_header = self.read_data_header(data)
        self._data = data[self.data_start:][:self.wave_header.data_length]
        self.len = self.wave_header.data_length // self.wave_header.channel_sample_width
        self.data_start = 0

    @property
    def is_thread_safe(self) -> bool:
        return self.thread_safe

    @classmethod
    def from_wav_data(cls, data: bytes, unit_sec: Optional[bool] = None) -> 'AudioSample':
        """
        Create an AudioSample object from a wav file data.
        Parameters
        ----------
        data : bytes
            The audio data as a bytes object.
        unit_sec : bool, optional
            Whether to manipulate in sample units or seconds units. If `unit_sec` is True, the indexes are treats a floating point seconds. If `unit_sec` is False, the indexes are treated as integer samples.
        """
        new = cls()
        new.unit_sec = unit_sec if unit_sec is not None else cls.unit_sec
        new.thread_safe = True
        new.wave_header = new.read_data_header(data)
        new._data = data[new.data_start:][:new.wave_header.data_length]
        new.len = new.wave_header.data_length // new.wave_header.channel_sample_width
        new.data_start = 0
        return new

    @classmethod
    def from_headerless_data(cls, data: bytes, sample_rate: int, precision=DEFAULT_PRECISION, channels=DEFAULT_CHANNELS, unit_sec=None):
        """
        Create an AudioSample object from audio data without a header.
        Parameters
        ----------
        data : bytes
            The audio data as a bytes object.
        sample_rate : int
            The sample rate of the audio data.
        precision : int, optional
            The bit depth of the audio data. If `precision` is None, the bit depth is set to DEFAULT_PRECISION (16).
        channels : int, optional
            The number of channels in the audio data. If `channels` is None, the number of channels is set to 1.
        unit_sec : bool, optional
            Whether to manipulate in sample units or seconds units. If `unit_sec` is True, the indexes are treats a floating point seconds. If `unit_sec` is False, the indexes are treated as integer samples.
        """
        new = cls()
        new.unit_sec = unit_sec if unit_sec is not None else cls.unit_sec
        new._data = data
        new.wave_header = WaveHeader(b"RIFF", WAVE_HEADER_LEN - RIFF_HEADER_LEN + len(data), b"WAVE", b"fmt ", FORMAT_HEADER_FULL_LEN, 1, channels,
                                     sample_rate, sample_rate * channels * (precision // 8), channels * (precision // 8), precision, b"\x00"*24, b"data",
                                     len(data))
        new.len = new.wave_header.data_length // new.wave_header.channel_sample_width
        new.data_start = 0
        return new

    def cleanup(self):
        """
        Clean up the AudioSample object
        - Closes the file handle if it exists.
        - Deletes the audio data if it exists.
        - Deletes the input container if it exists.
        """
        del self._data
        if self.input_container:
            del self.input_container
            self.input_container = None
        if self.f and getattr(self.f, 'close', None):
            if sys.getrefcount(self.f) <= 2:
                self.f.close()
            del self.f
            del self.wave_header

        self.f = None
        self.len = 0

    def close(self):
        if sys.getrefcount(self.f) <= 2:
            self.f.close()        
        self.f = None

    def write_sample_to_file(self, audio_path: Union[Path, str, io.BytesIO], no_encode=False):
        """
        Write the audio sample to a file.
        Parameters
        ----------
        audio_path : Union[Path, str, io.BytesIO]
            The path to write the audio sample to. If `audio_path` is a Path or str, the audio sample is written to the specified file. If `audio_path` is a io.BytesIO object, the audio sample is written to the io.BytesIO object.
        no_encode : bool, optional
            Whether to encode the audio data. If `no_encode` is True, the audio data is written as is. If `no_encode` is False, the audio data is encoded or re-encoded.
        """

        self.write_to_file(audio_path, no_encode)

    def write_to_file(self, audio_path: Union[Path, str, io.BytesIO], no_encode=False, force_out_format=None):
        """
        Write the audio sample to a file.
        Parameters
        ----------
        audio_path : Union[Path, str, io.BytesIO]
            The path to write the audio sample to. If `audio_path` is a Path or str, the audio sample is written to the specified file. If `audio_path` is a io.BytesIO object, the audio sample is written to the io.BytesIO object.
        no_encode : bool, optional
            Whether to encode the audio data. If `no_encode` is True, the audio data is written as is. If `no_encode` is False, the audio data is encoded or re-encoded.
        force_out_format : str, optional
            The format to encode the audio data to. If `force_out_format` is None, the audio data is written as is. If `force_out_format` is not None, the audio data is encoded to the specified format.
        """

        logger.info(f"Writing audio sample: {audio_path}")
        ACCEPTED_SUFFIXES = ["wav", "mp4", "m4a", "ogg", "mp3", "opus", "mov", "ts", "aac", "adts"]
        #TODO: support writing directly to file object.
        if force_out_format and not force_out_format in ACCEPTED_SUFFIXES:
            raise NotImplementedError("Not supported output format")

        if getattr(audio_path, 'seek', None) and getattr(audio_path, 'write', None):
            if getattr(audio_path, 'mode',None) and not "b" in audio_path.mode:
                raise ValueError("Unable to write non-binary file")
            self.as_data(out_file=audio_path, no_encode=no_encode, force_out_format=force_out_format)
            return
        if not force_out_format:
            if Path(audio_path).suffix[1:] in ACCEPTED_SUFFIXES:
                force_out_format = Path(audio_path).suffixes[-1][1:]
            else:
                raise ValueError("Either use supported force_out_format or specify a file extension")
        with open(str(audio_path), 'wb') as f:
            self.as_data(out_file=f, no_encode=no_encode, force_out_format=force_out_format)

    def write(self, audio_path: Union[Path, str, io.BytesIO], no_encode=False, force_out_format=None):
        """
        Write the audio sample to a file.
        Parameters
        ----------
        audio_path : Union[Path, str, io.BytesIO]
            The path to write the audio sample to. If `audio_path` is a Path or str, the audio sample is written to the specified file. If `audio_path` is a io.BytesIO object, the audio sample is written to the io.BytesIO object.
        no_encode : bool, optional
            Whether to encode the audio data. If `no_encode` is True, the audio data is written as is. If `no_encode` is False, the audio data is encoded or re-encoded.
        force_out_format : str, optional
            The format to encode the audio data to. If `force_out_format` is None, the audio data is written as is. If `force_out_format` is not None, the audio data is encoded to the specified format.
        """
        self.write_to_file(audio_path, no_encode, force_out_format)

    def save(self, audio_path: Union[Path, str, io.BytesIO]):
        """
        Write the audio sample to a file.
        Parameters
        ----------
        audio_path : Union[Path, str, io.BytesIO]
            The path to write the audio sample to. If `audio_path` is a Path or str, the audio sample is written to the specified file. If `audio_path` is a io.BytesIO object, the audio sample is written to the io.BytesIO object.
        """
        self.write_to_file(audio_path)



    
