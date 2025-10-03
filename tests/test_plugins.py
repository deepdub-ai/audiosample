import time
import os, subprocess, io
import pytest
import numpy as np
# import whisper

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


def test_audiosample_as_input(data_dir):
    a = AudioSample().beep(1).to_stereo()
    print(a.channels)
    tmp_file = data_dir / "tmp.wav"
    a.write(tmp_file)
    assert a.channels == 2
    assert AudioSample(a, force_channels=1).as_numpy().shape[0] == 2
    a = AudioSample(tmp_file)
    assert a.channels == 2
    assert AudioSample(a, force_channels=1).as_numpy().shape[0] == 2
    tmp_flac = data_dir / "tmp.flac"
    AudioSample(tmp_file).write(tmp_flac, force_out_format="flac")
    a = AudioSample(tmp_flac)
    assert a.channels == 2
    assert AudioSample(a, force_channels=1).as_numpy().shape[0] == 2
    tmp_mp3 = data_dir / "tmp.mp3"
    a.write(tmp_mp3)
    a = AudioSample(tmp_mp3)
    assert a.channels == 2
    assert AudioSample(a, force_channels=1).as_numpy().shape[0] == 2
