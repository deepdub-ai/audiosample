import numpy as np
from numba import njit, prange


@njit
def polyphase_downsample(filtered, decimation_factor):
    N = filtered.shape[0]
    # Estimate number of output samples
    num_out = int(N / decimation_factor)
    output = np.empty(num_out, dtype=filtered.dtype)
    pos = 0.0
    for i in range(num_out):
        idx = int(pos)
        frac = pos - idx
        # Make sure idx is within bounds
        if idx < 0:
            idx = 0
        elif idx >= N - 1:
            idx = N - 2
        output[i] = filtered[idx] * (1 - frac) + filtered[idx + 1] * frac
        pos += decimation_factor
    return output

@njit
def mu_law_encode(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """
    Encode a float32/float64 signal in [-1,1] to 8-bit mu-law.
    Returns a np.uint8 array in [0, 255].
    """
    n = audio.shape[0]
    result = np.empty(n, dtype=np.uint8)
    log_mu = np.log1p(mu)
    
    for i in prange(n):
        # 1) Clip the value to [-1, 1]
        x = audio[i]
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        
        # 2) Apply mu-law transformation:
        # Compute the magnitude component.
        magnitude = np.log1p(mu * np.abs(x)) / log_mu
        
        # Determine the sign.
        if x > 0.0:
            s = 1.0
        elif x < 0.0:
            s = -1.0
        else:
            s = 0.0
        
        # Get the mu-law encoded signal in [-1, 1]
        signal = s * magnitude
        
        # 3) Scale and convert to an 8-bit value in [0,255]
        # Multiplying by 127.5 maps [-1,1] to [0,255]
        result[i] = np.uint8((signal + 1.0) * 127.5)
    
    return result

class Downsampler:
    
    @staticmethod
    def _design_lowpass_filter(cutoff_hz: float, fs: float, numtaps: int=101, window='hamming') -> np.ndarray:
        """
        Create a windowed-sinc FIR filter for low-pass.
        
        cutoff_hz: cutoff frequency in Hz
        fs: sampling rate in Hz
        numtaps: number of FIR filter taps
        window: 'hamming' or 'hann' or 'blackman' etc. 
        """
        # Normalized cutoff (relative to Nyquist = fs/2)
        nyquist = fs / 2.0
        cutoff = cutoff_hz / nyquist  # between 0 and 1
        
        # Ideal sinc filter in time domain
        # We'll center it so "middle" is at (numtaps-1)/2
        n = np.arange(numtaps)
        center = (numtaps - 1) / 2.0

        # Avoid divide-by-zero with np.where or a small offset
        # sinc(x) = sin(pi*x)/(pi*x). In discrete-time, x = (n - center)*cutoff
        h = np.sinc(cutoff * (n - center))
        
        # Apply a window (e.g. Hamming)
        if window == 'hamming':
            w = np.hamming(numtaps)
        elif window == 'hann':
            w = np.hanning(numtaps)
        elif window == 'blackman':
            w = np.blackman(numtaps)
        else:
            raise ValueError("Unsupported window type.")
        
        h *= w
        
        # Normalize filter so its sum = 1.0 (unity gain at DC)
        h /= np.sum(h)
        
        print("INIT")
        return h
    
    def __init__(self, orig_sr: int, target_sr: int, numtaps: int=101, window='hamming'):
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.decimation_factor = orig_sr / target_sr
        self.ceil_decimation_factor = np.ceil(self.decimation_factor)
        self.numtaps = numtaps
        self.window = window
        self.min_input_buffer_size = self.numtaps + self.ceil_decimation_factor+1

        self.remaining_samples = np.array([])

        self.padding_size = int(np.ceil(self.numtaps / self.decimation_factor)) * int(np.ceil(self.decimation_factor))
        self.total_num_input_samples = 0
        self.total_num_output_samples = 0

        self.fir = self._design_lowpass_filter(cutoff_hz=target_sr/2, fs=orig_sr, numtaps=numtaps, window=window)


    def encode(self, audio_chunk: np.ndarray=None) -> np.ndarray:
        """
        Downsample from 48 kHz to target_sr using a basic FIR low-pass filter + decimation.
        Handles streaming input by maintaining filter state between calls.
        
        Args:
            audio: Input audio chunk. If None, flushes remaining samples.

        Returns:
            Downsampled audio chunks
        """
        reset_params_when_done = False
        if audio_chunk is None:
            # Add enough padding to process remaining samples
            self.remaining_samples = np.concatenate([self.remaining_samples, np.zeros(self.padding_size)])
            reset_params_when_done = True
        else:
            if audio_chunk.ndim != 1:
                raise ValueError("audio must be 1D")
            # Concatenate with remaining samples from previous chunk
            self.remaining_samples = np.concatenate([self.remaining_samples, audio_chunk])
        
        if self.remaining_samples.shape[-1] < self.min_input_buffer_size:
            if reset_params_when_done:
                self.__init__(self.orig_sr, self.target_sr, self.numtaps, self.window)
            return np.array([])
        # Apply FIR filter
        filtered = np.convolve(self.remaining_samples, self.fir, mode='valid')
        num_output_samples = int((self.total_num_input_samples + len(filtered)) / self.decimation_factor)
        if num_output_samples - self.total_num_output_samples == 0:
            if reset_params_when_done:
                self.__init__(self.orig_sr, self.target_sr, self.numtaps, self.window)
            return np.array([])
        sample_positions = np.arange(self.total_num_output_samples, num_output_samples)*self.decimation_factor
        sample_positions_int = sample_positions.astype(np.int32)
        
        # Linear interpolation between samples
        frac = sample_positions - sample_positions_int
        orig_sample_positions_int = sample_positions_int
        sample_positions_int = sample_positions_int - self.total_num_input_samples
        
        audio_target_sr = filtered[sample_positions_int] * (1 - frac) + \
                          filtered[np.minimum(sample_positions_int + 1, len(filtered) - 1)] * frac
        
        self.total_num_output_samples += len(audio_target_sr)
        self.remaining_samples = self.remaining_samples[sample_positions_int[-1]+1:]

        self.total_num_input_samples = orig_sample_positions_int[-1]+1



        if reset_params_when_done:
            self.__init__(self.orig_sr, self.target_sr, self.numtaps, self.window)
        return audio_target_sr