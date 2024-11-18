from typing import Union
import numpy as np
from audiosample import AudioSample
from logging import getLogger
logger = getLogger(__name__)

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

def mix(self, start, other, fade_duration=None, clipping_strategy=None):
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
    clipping_strategy: None or str
        The strategy to use when clipping occurs. The following strategies are available:
        - None: Clipping is not handled.
        - 'norm': Normalize post-mixing to avoid clipping.
        - 'normif': Normalize post-mixing only if clipping occurs.
        - 'warn': Warn if there is clipping
        - 'raise': Raise an exception if there is clipping

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
    if clipping_strategy:
        if clipping_strategy == 'norm':
            out = out / np.max(np.abs(out))
        elif clipping_strategy == 'normif':
            if np.max(np.abs(out)) > 1:
                out = out / np.max(np.abs(out))
        elif clipping_strategy == 'warn':
            if np.max(np.abs(out)) > 1:
                logger.warning("Clipping occurred during mixing")
        elif clipping_strategy == 'raise':
            if np.max(np.abs(out)) > 1:
                raise ValueError("Clipping occurred during mixing")
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

def stretch(self, tempo):
    """
    Time-stretch an audio signal without changing its pitch using the WSOLA algorithm.
    Adjusted to handle multi-channel input where the first dimension is channels.
    
    Parameters:
    - tempo: float
        The desired tempo change factor (e.g., 1.5 for 50% faster).
    
    Returns:
    - output_audio: AudioSample
        The time-stretched audio signal as a NumPy array with the same shape as input.
    """
    assert tempo > 0, "Tempo must be greater than 0"
    audio = self.as_numpy()
    sample_rate = self.sample_rate
    audio = self.as_numpy()
    sample_rate = self.sample_rate
    # Ensure audio is at least 2D
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    channels, num_samples = audio.shape
    
    # Parameters (milliseconds)
    overlap_ms = 12     # Length of the overlapping window
    
    # Adjust sequence_ms and seek_window_ms based on tempo
    AUTOSEQ_TEMPO_LOW = 0.5
    AUTOSEQ_TEMPO_TOP = 2.0

    AUTOSEQ_AT_MIN = 125.0
    AUTOSEQ_AT_MAX = 50.0
    AUTOSEQ_K = (AUTOSEQ_AT_MAX - AUTOSEQ_AT_MIN) / (AUTOSEQ_TEMPO_TOP - AUTOSEQ_TEMPO_LOW)
    AUTOSEQ_C = AUTOSEQ_AT_MIN - (AUTOSEQ_K) * (AUTOSEQ_TEMPO_LOW)

    AUTOSEEK_AT_MIN = 25.0
    AUTOSEEK_AT_MAX = 15.0
    AUTOSEEK_K = (AUTOSEEK_AT_MAX - AUTOSEEK_AT_MIN) / (AUTOSEQ_TEMPO_TOP - AUTOSEQ_TEMPO_LOW)
    AUTOSEEK_C = AUTOSEEK_AT_MIN - (AUTOSEEK_K) * (AUTOSEQ_TEMPO_LOW)

    def check_limits(x, mi, ma):
        return max(mi, min(x, ma))

    seq = AUTOSEQ_C + AUTOSEQ_K * tempo
    seq = check_limits(seq, AUTOSEQ_AT_MAX, AUTOSEQ_AT_MIN)
    sequence_ms = int(seq + 0.5)

    seek = AUTOSEEK_C + AUTOSEEK_K * tempo
    seek = check_limits(seek, AUTOSEEK_AT_MAX, AUTOSEEK_AT_MIN)
    seek_window_ms = int(seek + 0.5)

    # Convert to samples
    sequence_length = int(sample_rate * sequence_ms / 1000)
    seek_window_length = int(sample_rate * seek_window_ms / 1000)
    overlap_length = int(sample_rate * overlap_ms / 1000)

    # Ensure lengths are even and positive
    sequence_length = max(2, sequence_length - (sequence_length % 2))
    seek_window_length = max(2, seek_window_length - (seek_window_length % 2))
    overlap_length = max(2, overlap_length - (overlap_length % 2))

    # Ensure seek_window_length >= 2 * overlap_length
    if seek_window_length < 2 * overlap_length:
        seek_window_length = 2 * overlap_length

    # Initialize buffers
    input_buffer = audio.copy()
    output_buffer = []
    p_mid_buffer = np.zeros((channels, overlap_length), dtype=np.float32)

    # Calculate nominal skip and sample requirement
    nominal_skip = tempo * (seek_window_length - overlap_length)
    position = 0
    skip_fract = 0.0

    # Precompute fade-in and fade-out envelopes
    fade_in = np.linspace(0, 1, overlap_length)[np.newaxis, :]
    fade_out = 1.0 - fade_in

    # Main processing loop
    while position + seek_window_length <= num_samples:
        ref_pos = position

        # Extract seek window
        seek_window = input_buffer[:, ref_pos:ref_pos + seek_window_length]

        # Vectorize cross-correlation over all possible offsets
        offsets = np.arange(0, seek_window_length - overlap_length + 1)
        num_offsets = offsets.shape[0]

        # Extract all possible segments for overlap
        segments_shape = (channels, num_offsets, overlap_length)
        segments_strides = (
            input_buffer.strides[0],
            input_buffer.strides[1],
            input_buffer.strides[1]
        )
        segments = np.lib.stride_tricks.as_strided(
            seek_window,
            shape=segments_shape,
            strides=segments_strides
        )

        # Flatten segments and p_mid_buffer for computation
        segments_flat = segments.reshape(channels, num_offsets, overlap_length)
        p_mid_buffer_flat = p_mid_buffer

        # Compute cross-correlation for all offsets
        corr = np.einsum('cno,co->n', segments_flat, p_mid_buffer_flat)
        norm_seg = np.sqrt(np.sum(segments_flat ** 2, axis=(0, 2)))
        norm_mid = np.sqrt(np.sum(p_mid_buffer_flat ** 2))

        # Avoid division by zero
        denom = norm_mid * norm_seg
        denom[denom == 0] = 1e-9

        correlations = corr / denom

        # Apply heuristic weighting
        tmp = (2 * offsets - (seek_window_length - overlap_length)) / (seek_window_length - overlap_length)
        weight = 1.0 - 0.25 * tmp ** 2
        correlations *= weight

        # Find best offset
        best_index = np.argmax(correlations)
        best_offset = offsets[best_index]

        # Overlap and add
        offset = best_offset
        overlap_segment = input_buffer[:, ref_pos + offset:ref_pos + offset + overlap_length]

        # Ensure overlap_segment has correct shape
        if overlap_segment.shape[1] < overlap_length:
            pad_length = overlap_length - overlap_segment.shape[1]
            padding = np.zeros((channels, pad_length), dtype=overlap_segment.dtype)
            overlap_segment = np.hstack([overlap_segment, padding])

        overlapped = p_mid_buffer * fade_out + overlap_segment * fade_in
        output_buffer.append(overlapped)

        # Calculate non-overlapping part length
        nominal_skip_int = int(nominal_skip + 0.5)
        non_overlap_length = seek_window_length - overlap_length - best_offset - overlap_length
        if non_overlap_length < 0:
            non_overlap_length = 0

        # Add non-overlapping part
        start = ref_pos + best_offset + overlap_length
        end = start + non_overlap_length
        if end > num_samples:
            end = num_samples
        if start < end:
            output_buffer.append(input_buffer[:, start:end])

        # Update p_mid_buffer
        position_increment = best_offset + overlap_length + non_overlap_length
        position += position_increment

        # Update p_mid_buffer for next iteration
        p_mid_buffer = input_buffer[:, position:position + overlap_length]
        if p_mid_buffer.shape[1] < overlap_length:
            pad_length = overlap_length - p_mid_buffer.shape[1]
            padding = np.zeros((channels, pad_length), dtype=p_mid_buffer.dtype)
            p_mid_buffer = np.hstack([p_mid_buffer, padding])

        # Adjust position based on tempo
        skip_fract += nominal_skip - position_increment
        if skip_fract >= 1.0:
            position += int(skip_fract)
            skip_fract -= int(skip_fract)
        elif skip_fract <= -1.0:
            position -= int(-skip_fract)
            skip_fract += int(-skip_fract)

        # Prevent infinite loops
        if position_increment == 0:
            position += 1  # Ensure progress

    # Append any remaining samples
    if position < num_samples:
        output_buffer.append(input_buffer[:, position:])

    # Concatenate output along samples axis
    output_audio = np.hstack(output_buffer)
    output_audio = np.clip(output_audio, -1.0, 1.0)

    return AudioSample.from_numpy(output_audio, self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('stretch', stretch)

def pan(self, pan: Union[float, str]):
    """
    Apply a stereo panning effect to the audio sample.
    Parameters:
    - pan: float or str
        The pan value between -1 (left) and 1 (right). Alternatively, 'left' or 'right' can be used.
    """
    if isinstance(pan, str):
        if pan == 'left':
            pan = -1
        elif pan == 'right':
            pan = 1
        else:
            raise ValueError("Invalid pan value. Must be a float between -1 and 1 or 'left' or 'right'")
    assert -1 <= pan <= 1, "Pan value must be between -1 and 1"
    assert self.channels == 2, "Stereo panning only supported for stereo audio"
    pan = (pan + 1) / 2
    left = np.sqrt(1 - pan)
    right = np.sqrt(pan)
    n = self.as_numpy()
    n[0] *= left
    n[1] *= right
    return AudioSample.from_numpy(n, rate=self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('pan', pan)

def to_robot(self, modulation_frequency=50, pitch_shift_semitones=-5):
    """
    Apply a robot voice effect to the audio sample by modulating the signal with a sine wave and shifting the pitch.
    Parameters:
    - modulation_frequency: int
        The frequency of the sine wave modulation.
    - pitch_shift_semitones: int
        The pitch shift in semitones.
    Returns:
    - output_audio: AudioSample
        The audio signal with the robot voice effect.
    """
    def resample(x, num):
        """
        Resample a signal to a different number of samples using the Fourier method.
    
        Parameters:
        x : array_like
            The original signal to be resampled.
        num : int
            The number of samples in the resampled signal.
    
        Returns:
        y : ndarray
            The resampled signal.
        """
        N = len(x)
        X = np.fft.fft(x)
        X = np.fft.fftshift(X)
    
        # Calculate the resampling factor
        ratio = float(num) / N
    
        if num < N:
            # Truncate the spectrum
            start = (N - num) // 2
            X_resampled = X[start:start+num]
        else:
            # Pad the spectrum with zeros
            pad_size = num - N
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            X_resampled = np.pad(X, (pad_left, pad_right), mode='constant', constant_values=0)
    
        X_resampled = np.fft.ifftshift(X_resampled)
        y = np.fft.ifft(X_resampled) * ratio
        return y.real

    def apply_sine_modulation(data, rate, frequency=10):
        t = np.arange(data.shape[-1]) / rate
        modulation = np.sin(2 * np.pi * frequency * t)
        return data * modulation
    
    def pitch_shift(data, rate, shift):
        factor = 2 ** (shift / 12.0)
        num_samples = int(data.shape[-1] / factor)
        resampled_data = resample(data, num_samples)
        return resample(resampled_data, len(data))
    
    assert self.channels == 1, "Robot voice effect only supported for mono audio"
    assert modulation_frequency > 0, "Modulation frequency must be greater than 0"

    rate, data = self.sample_rate, self.as_numpy()
    # Normalize data
    max_val = np.max(np.abs(data))
    data = data / max_val
    
    # Apply sine modulation
    modulated_data = apply_sine_modulation(data, rate, modulation_frequency)
    
    # Apply pitch shift
    shifted_data = pitch_shift(modulated_data, rate, pitch_shift_semitones)
    
    # Normalize the data back to the original range
    shifted_data = shifted_data / np.max(np.abs(shifted_data)) * max_val

    return AudioSample.from_numpy(shifted_data, rate=self.sample_rate, precision=self.precision, unit_sec=self.unit_sec)

AudioSample.register_plugin('to_robot', to_robot)