# AudioSample 
#### by [deepdub.ai](https://deepdub.ai/)
AudioSample is an optimized numpy-like audio manipulation library, created for researchers, used by developers.

It is an advanced audio manipulation library designed to provide researchers and developers with efficient, numpy-like tools for audio processing. It supports complex audio operations with ease and offers a familiar syntax for those accustomed to numpy.

## Features

- **Seamless Audio Operations:** Perform a wide range of audio manipulations, including mixing, filtering, and transformations.
- **Integration with Numpy:** Leverage numpy's syntax and capabilities for intuitive audio handling.
- **Integration with Torch:** Export audio directly to and from torch tensors.
- **High Performance:** Optimized for speed and efficiency, suitable for research and production environments. Most actions are lazy, so no operation done until absolutely necessary.
- **Extensive I/O Support:** Easily read from and write to various audio formats. Utilizes PyAv - to support multiple ranges.

## Installation

To install AudioSample, use pip:

to install all prerequisites:
```bash
pip install audiosample[all] 
#Possible extras are:

[av] - only av
[torch] - add torch
[tests] - include everything for tests.
[noui] - install without jupyter support.
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

- **FFT Analysis:** Perform fast Fourier transforms to analyze frequency components.
- **Normalization:** Easily normalize audio levels.
- **Mixing:** Easily mix multiple audio sources together. Using * sign
- **Concat** Easily concat audio sources. Using + sign
- **Playback:** Play audio directly in Jupyter notebooks or from the command line.
## Documentation

For detailed instructions and API references, type help(AudioSample)

## Examples

Explore the [examples notebook](examples.ipynb) to see practical applications of AudioSample in action.

## LICENSE

AudioSample is released under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please follow the [contributing guidelines](CONTRIBUTING.md) to submit changes.

## About Deepdub

AudioSample is developed by [Deepdub](https://deepdub.ai/), a company specializing in AI-driven audio solutions. Deepdub focuses on enhancing media experiences through cutting-edge technology, enabling content creators to reach global audiences with high-quality, localized audio.

## Support

If you have questions or need help, please open an issue on GitHub.
