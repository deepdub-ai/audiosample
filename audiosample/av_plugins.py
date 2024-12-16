from audiosample import AudioSample

def to_stereo(self):
    """Convert mono audio to stereo. If the AudioSample is already stereo, the original sample is returned. If the AudioSample is not already in memory, the conversion is done on the fly."""
    if self.channels == 2:
        return self
    if self.channels != 1:
        raise ValueError("Can only convert mono to stereo")
    if not getattr(self, '_data', None):
        #def __init__(self, f=None, force_read_format=None, force_sample_rate=None, force_channels=None, force_precision=None, unit_sec=False, thread_safe=False, stream_idx=0):
        new = self.__class__(self.f, force_read_format=self.force_read_format, force_sample_rate=self.force_sample_rate, 
                             force_channels=2, force_precision=self.force_precision, 
                             unit_sec=self.unit_sec, thread_safe=self.thread_safe, stream_idx=self.stream_idx)
        new.data_start = self.data_start
        new.len = self.len
    else:
        new = self.__class__(self.as_data(), force_channels=2, unit_sec=self.unit_sec)
    return new

AudioSample.register_plugin('to_stereo', to_stereo)

def to_mono(self):
    """Convert stereo audio to mono. If the AudioSample is already mono, the original sample is returned. If the AudioSample is not already in memory, the conversion is done on the fly."""
    if self.channels == 1:
        return self
    if self.channels != 2:
        raise ValueError("Can only convert stereo to mono")
    if not getattr(self, '_data', None):
        self.f.seek(0,0)
        new = self.__class__(self.f, force_read_format=self.force_read_format, force_sample_rate=self.force_sample_rate, 
                             force_channels=1, force_precision=self.force_precision, 
                             unit_sec=self.unit_sec, thread_safe=self.thread_safe, stream_idx=self.stream_idx)
        new.start = self.start
        new.len = self.len
    else:
        new = self.__class__(self.as_data(), force_channels=1, unit_sec=self.unit_sec)
    return new

AudioSample.register_plugin('to_mono', to_mono)

def resample(self, sample_rate):
    """Resample audio to a different sample_rate. If the AudioSample is already at the desired sample_rate, the original
    sample is returned. If the AudioSample is not already in memory, the resampling is done on the fly.
    """
    if self.sample_rate == sample_rate:
        return self
    if not getattr(self, '_data', None):
        self.f.seek(0,0)
        new = self.__class__(self.f, force_read_format=self.force_read_format, force_sample_rate=sample_rate, 
                             force_channels=self.channels, force_precision=self.force_precision, 
                             unit_sec=self.unit_sec, thread_safe=self.thread_safe, stream_idx=self.stream_idx)
        new.start = int(sample_rate*self.start/self.sample_rate)
        new.len = int(sample_rate*self.len/self.sample_rate)
    else:
        new = self.__class__(self.as_data(), force_sample_rate=sample_rate, unit_sec=self.unit_sec)
    return new

AudioSample.register_plugin('resample', resample)