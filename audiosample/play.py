import io
from contextlib import redirect_stderr
from audiosample import AudioSample
import ctypes
import time
try:
    import pyaudio
    LINUX_PLAY_SUPPORTED = True
except ImportError:
    #warnings.warn("AudioSample unable to support audio playing, please install pyaudio")
    LINUX_PLAY_SUPPORTED = False


GLOBAL_PLAYER = None

IGNORE_ASOUND_ERRORS = False
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
def ignore_asound_errors():
    global IGNORE_ASOUND_ERRORS
    if IGNORE_ASOUND_ERRORS:
        return
    IGNORE_ASOUND_ERRORS = True

    asound = ctypes.cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)



def play(self, global_init=True):
    """
    Play the audio sample. This uses pyaudio.
    Data is first loaded into memory.
    """
    if not LINUX_PLAY_SUPPORTED:
        raise NotImplementedError("play() not supported on this platform")
    p = None
    if global_init:
        global GLOBAL_PLAYER
        if not GLOBAL_PLAYER:
            ignore_asound_errors()
            GLOBAL_PLAYER = pyaudio.PyAudio()
        p = GLOBAL_PLAYER
    else:
        ignore_asound_errors()
        p = pyaudio.PyAudio()
    self.play_err = io.StringIO()

    stream = p.open(format=p.get_format_from_width(self.sample_width),
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True)
    
    if self.f and not self._data and not self.iterable_input_buffer:
        self.read()
    try:
        if self.iterable_input_buffer:
            for chunk in self.as_data_stream(force_out_format='s16le'):
                stream.write(chunk)
                time.sleep(0.005)
        else:
            chunk_size = self.sample_rate // 10
            for chunk_start in range(0, len(self), chunk_size):
                stream.write(self[chunk_start:chunk_start + chunk_size]._data)
    finally:
        # let the final chunk playout before stopping the stream.
        #time.sleep(0.25)
        stream.stop_stream()
        stream.close()
        if not global_init:
            p.terminate()

AudioSample.register_plugin('play', play)