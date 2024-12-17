# AudioSample 
#### by [deepdub.ai](https://deepdub.ai/)
AudioSample is an optimized numpy-like audio manipulation library, created for researchers, used by developers.

It is an advanced audio manipulation library designed to provide researchers and developers with efficient, numpy-like tools for audio processing. It supports complex audio operations with ease and offers a familiar syntax for those accustomed to numpy.

AudioSample is perfect for data loading and ETLs, because its fast and has a low memory footprint due to lazy actions.

## Features

- **Seamless Audio Operations:** Perform a wide range of audio manipulations, including mixing, filtering, and transformations.
- **Integration with Numpy:** Leverage numpy's syntax and capabilities for intuitive audio handling.
- **Integration with Torch:** Export audio directly to and from torch tensors.
- **High Performance:** Optimized for speed and efficiency, suitable for research and production environments. Most actions are lazy, so no operation done until absolutely necessary.
- **Extensive I/O Support:** Easily read from and write to various audio formats. Utilizes PyAv - to support multiple ranges.

## Release notes 2.2.1
- Support up to numpy 2.2.0
- Streaming input, streaming output:
   - AudioSample now supports receiving a python generator for input Generator[Union[bytes,numpy,AudioSample]]
   - Warning: It currently still stores everything in memory so this can't live forever.
   - Plugin functionality is not supported in stream mode.
   - streaming mode requires PyAV (See example below):
- Constructor supports numpy buffers (same as calling AudioSample.from_numpy use force_read_sample_rate to set sample rate.)

## Installation

To install AudioSample, use pip:

to install all prerequisites:
```bash
pip install audiosample[all] 
#linux/WSL:
pip install audiosample[all] 

#Possible extras are:
[av] - only av
[torch] - add torch
[tests] - include everything for tests.
[noui] - install without jupyter support.

#Mac OS:
brew install portaudio
#linux/WSL:
apt-get install portaudio19-dev
[play] - bare, with ability to play audio in console. (uses pyaudio)
```



## Usage

Here's a quick example of how to load, process, and save audio using AudioSample:

```python
import audiosample as ap
import numpy as np

# Create a 1 second audio sample with 44100 samples per second and 2 channels
au = ap.AudioSample.from_numpy(np.random.rand(2, 48000), rate=48000)
beep = ap.AudioSample().beep(1).to_stereo()
out = au.gain(-12) * beep
out.write("beep_with_overlayed_noise.mp3")
out = au.gain(-10) + au.silence(1) + beep
out.write("noise_then_silence_then_beep.mp3")

```

### Additional Operations
- **Resampling:** Fast resampling of audio.
- **Normalization:** Easily normalize audio levels.
- **Mixing:** Easily mix multiple audio sources together. Using * sign
- **Concat** Easily concat audio sources. Using + sign
- **Playback:** Play audio directly in Jupyter notebooks or from the command line.
## Documentation

## Bench Marks

### AudioSample outperforms PyDub

### open concatenation and save.
- longbeep is a 100s long wav file of beep.

```python
import pydub
from audiosample import AudioSample
def test_audiosample():
    au = AudioSample()
    for i in range(0, 100):
        au += AudioSample("longbeep.wav")[50:51]
    au.write("out.wav")

def test_pydub():
    au = pydub.AudioSegment.empty()
    for i in range(0, 100):
        au += pydub.AudioSegment.from_file("longbeep.wav")[50:51]
    au.export("out.wav")

%timeit test_audiosample()
#52.9 ms ± 1.89 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit test_pydub()
#376 ms ± 15.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

### AudioSample mix vs. PyDub overlay
```python
def test_audiosample():
    au = AudioSample().silence(1)
    for i in range(0, 100):
        au *= AudioSample("longbeep.wav")[50:51]
    au.write("out.wav")
def test_pydub():
    au = pydub.AudioSegment.silent(1)
    for i in range(0, 100):
        au = au.overlay(pydub.AudioSegment.from_file("longbeep.wav")[50:51], 0)
    au.export("out.wav")

In [3]: %timeit test_audiosample()
12.7 ms ± 265 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
In [4]: %timeit test_pydub()
398 ms ± 26.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

### AudioSample outperforms SoundFile

verylongbeep.wav - is a 3200s file. (293M)
```python
import soundfile as sf
from audiosample import AudioSample

def test_audiosample():
    out = AudioSample("verylongbeep.wav")[1500:1501].as_numpy()

def test_soundfile():
    with sf.SoundFile("verylongbeep.wav") as f:
        f.seek(48000*1500)
        out = f.read(48000)

In [5]: %timeit test_audiosample()
35.8 μs ± 1.69 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
In [6]: %timeit test_soundfile()
140 μs ± 8.89 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```


For detailed instructions and API references, type help(AudioSample)

## Examples

Explore the [examples notebook](examples.ipynb) to see practical applications of AudioSample in action.

### Streaming code example below:
```

def chunkify(buffer: bytes):
    CHUNK_SIZE = 1000
    for i in range(0, len(buffer), CHUNK_SIZE):
        yield buffer[i:i+CHUNK_SIZE]

testmp3 = open('test.mp3','rb').read()

collect = b''
for chunk in AudioSample(chunkify(testmp3), force_read_sample_rate=48000, force_sample_rate=8000).as_data_stream(force_out_format='mulaw')
     collect += chunk

open('test.mulaw','wb').write(chunk)

```


## License

AudioSample is released under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please follow the [contributing guidelines](CONTRIBUTING.md) to submit changes.

## About Deepdub

AudioSample is developed by [Deepdub](https://deepdub.ai/), a company specializing in AI-driven audio solutions. Deepdub focuses on enhancing media experiences through cutting-edge technology, enabling content creators to reach global audiences with high-quality, localized audio.

## Support

If you have questions or need help, please open an issue on GitHub.
