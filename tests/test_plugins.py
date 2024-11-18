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

def test_mix(data_dir, small_wav_file):
    a = AudioSample(small_wav_file)
    b = AudioSample(small_wav_file)
    c = a * b
    assert c.channels == 2
    assert c.sample_rate == a.sample_rate
    assert c.len == a.len
    assert c.precision == a.precision
    assert c.unit_sec == a.unit_sec

def test_pan(data_dir, small_wav_file):
    a = AudioSample(small_wav_file).to_stereo()
    b = a.pan(0.5)
    assert b.channels == 2
    assert b.sample_rate == a.sample_rate
    assert b.len == a.len
    assert b.precision == a.precision
    assert b.unit_sec == a.unit_sec
    b = a.pan(-1)
    assert np.all(b.as_numpy() == a.pan("left").as_numpy())
    nb = b.as_numpy()
    assert nb.shape[0] == 2
    assert nb[0].sum() != 0
    assert nb[1].sum() == 0
    b = a.pan(1)
    assert np.all(b.as_numpy() == a.pan("right").as_numpy())
    nb = b.as_numpy()
    assert nb.shape[0] == 2
    assert nb[0].sum() == 0
    assert nb[1].sum() != 0
