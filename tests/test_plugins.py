import time
import os, subprocess, io
import pytest
import numpy as np
# import whisper
from scipy.signal import correlate

from audiosample import AudioSample

@pytest.fixture(scope='session')
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('audiosample_data')

@pytest.fixture(scope='session')
def small_wav_file(data_dir):
    return os.path.abspath(f'{__file__}/../assets/audio_files/test.wav')


def test_concat(data_dir, small_wav_file):
    a = AudioSample(small_wav_file)
    b = AudioSample(small_wav_file)
    c = a + b
    assert c.channels == 2
    assert c.sample_rate == a.sample_rate
    assert c.len == a.len + b.len
    assert c.precision == a.precision
    assert c.unit_sec == a.unit_sec

def test_concat_fail(data_dir, small_wav_file):
    a = AudioSample(small_wav_file)
    b = AudioSample(small_wav_file, force_channels=1)
    with pytest.raises(ValueError):
        c = a + b

def test_concat_fail2(data_dir, small_wav_file):
    a = AudioSample(small_wav_file)
    b = AudioSample(small_wav_file, force_sample_rate=a.sample_rate//2)
    with pytest.raises(ValueError):
        c = a + b

def test_mix(data_dir, small_wav_file):
    a = AudioSample(small_wav_file)
    b = AudioSample(small_wav_file)
    c = a * b
    assert c.channels == 2
    assert c.sample_rate == a.sample_rate
    assert c.len == a.len
    assert c.precision == a.precision
    assert c.unit_sec == a.unit_sec