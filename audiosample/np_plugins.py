import numpy as np
from audiosample import AudioSample

def mul(self, other):
    """
    Mix two AudioSamples together. The two AudioSamples must have the same sample rate and number of channels.
    """
    if not isinstance(other, AudioSample):
        raise ValueError("Can only multiply AudioSample with another AudioSample")
    if self.sample_rate != other.sample_rate:
        raise ValueError("Cannot multiply AudioSamples with different sample rates")
    if self.channels != other.channels:
        raise ValueError("Cannot multiply AudioSamples with different number of channels")
    return self.mix(0, other)

AudioSample.register_plugin('__mul__', mul)

def silence(self, duration, channels=None):
    """
    Generate silence for a given duration in seconds.
    """
    len_in_unit = int(duration * self.sample_rate)
    channels = channels or self.channels
    return AudioSample.from_numpy(np.zeros([channels, len_in_unit]), rate=self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('silence', silence)

def beep(self, duration, freq=1200.0, amplitude=1.0):
    """
    Generate a sine wave beep for a given duration in seconds.
    """
    assert amplitude > 0, "Amplitude must be bigger than 0"
    assert isinstance(duration, int) or isinstance(duration, float) and self.unit_sec, "Duration must be an integer or float with unit_sec=True"
    duration = int(duration * self.sample_rate) if self.unit_sec else duration
    t = np.linspace(0, duration / self.sample_rate, duration, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
    return AudioSample.from_numpy(sine_wave, rate=self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('beep', beep)
AudioSample.register_plugin('tone', beep)

def gain(self, gain_in_db, pre_normalize=False):
    """
    Apply gain to the audio sample. The gain is in decibels.
    """
    factor = 10 ** (gain_in_db / 20)
    n = self.as_numpy()
    if pre_normalize:
        n = n / np.max(np.abs(n))
    return AudioSample.from_numpy(n * factor, self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('gain', gain)

def mix(self, start, other, fade_duration=None):
    """
    Mix two AudioSamples together. The two AudioSamples must have the same sample rate and number of channels.
    Parameters:
    ----------
    start: int or float
        The start position in samples or seconds, within the first AudioSample, where the second AudioSample will be mixed in.
    other: AudioSample
        The second AudioSample to mix in.
    fade_duration: int or float or None
        The duration in samples or seconds, within the second AudioSample, over which the mix will fade in and out. If None, no fade is applied.
        Fading is used to avoid clicks when mixing audio samples.
    """
    assert fade_duration is None or fade_duration >= 0, "Fade in duration must be greater than or equal to 0"
    assert isinstance(start, int) or (self.unit_sec and isinstance(start, float)), "Start must be an integer or float with unit_sec=True"
    assert isinstance(fade_duration, int) or self.unit_sec and isinstance(fade_duration, float) or fade_duration is None, "Fade in duration must be an integer or float with unit_sec=True or None"

    if not isinstance(other, AudioSample):
        raise ValueError("Can only overlay AudioSample with another AudioSample")
    if self.sample_rate != other.sample_rate:
        raise ValueError("Cannot overlay AudioSamples with different sample rates")
    if self.channels != other.channels:
        raise ValueError("Cannot overlay AudioSamples with different number of channels")
    if self.unit_sec:
        start = int(start*self.sample_rate)
        fade_duration = int((fade_duration or 0) * self.sample_rate)
    if start < 0:
        raise ValueError("Start must be greater than or equal to 0")
    if start + len(other) > len(self):
        n = self.as_numpy()
        shape = [*n.shape]
        shape[-1] = start + len(other) - len(self)
        out = np.concatenate((n, np.zeros(shape)), axis=-1)
    else:
        out = self.as_numpy()
    to_mix_in = other.as_numpy()
    if fade_duration:
        to_mix_in[..., :fade_duration] *= np.linspace(0, 1, fade_duration)
        to_mix_in[..., -fade_duration:] *= np.linspace(1, 0, fade_duration)
    out[..., start:start + len(other)] += to_mix_in
    return AudioSample.from_numpy(out, self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('mix', mix)

def fadein(self, duration):
    """
    Apply a fade in effect to the audio sample. The duration is in the default units.
    """
    assert isinstance(duration, int) or isinstance(duration, float) and self.unit_sec, "Duration must be an integer or float with unit_sec=True"
    duration = int(duration * self.sample_rate) if self.unit_sec else duration
    if duration > len(self):
        raise ValueError("Fade in duration cannot be longer than the audio")
    n = self.as_numpy()
    n[..., :duration] *= np.linspace(0, 1, duration)
    return AudioSample.from_numpy(n, self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('fadein', fadein)

def fadeout(self, duration):
    """
    Apply a fade out effect to the audio sample. The duration is in the default units.
    """
    assert isinstance(duration, int) or isinstance(duration, float) and self.unit_sec, "Duration must be an integer or float with unit_sec=True"
    duration = int(duration * self.sample_rate) if self.unit_sec else duration
    if duration > len(self):
        raise ValueError("Fade out duration cannot be longer than the audio")
    n = self.as_numpy()
    n[..., -duration:] *= np.linspace(1, 0, duration)
    return AudioSample.from_numpy(n, self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('fadeout', fadeout)