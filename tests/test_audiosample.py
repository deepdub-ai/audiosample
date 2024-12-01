import time
import os, subprocess, io
import pytest
import multiprocessing.pool
import numpy as np
import pickle
from functools import partial
# import whisper
from scipy.signal import correlate

from audiosample import AudioSample

@pytest.fixture(scope='session')
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('audiosample_data')

@pytest.fixture(scope='session')
def small_mp3_file(data_dir):
    return os.path.abspath(f'{__file__}/../assets/audio_files/test.mp3')

@pytest.fixture(scope='session')
def short_mp3_file(data_dir, small_mp3_file):
    fname = os.path.abspath(f'{data_dir}/short_mp3_file.mp3')
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {small_mp3_file} -t 5 {fname}"])
    return fname


@pytest.fixture(scope='session')
def small_opus_file(data_dir):
    return os.path.abspath(f'{os.path.dirname(__file__)}/assets/audio_files/test-small.opus')

@pytest.fixture(scope='session')
def small_24bit_wav_file(data_dir):
    return os.path.abspath(f'{os.path.dirname(__file__)}/assets/audio_files/test-24bit.wav')

@pytest.fixture(scope='session')
def dual_stream_mp4_file(data_dir, small_mp3_file):
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {small_mp3_file} -ss 10 -t 10 -c copy {data_dir}/skip_10_sec.mp3"])
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {small_mp3_file} -t 20 -c copy {data_dir}/only_20_sec.mp3"])
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {data_dir}/only_20_sec.mp3 -i {data_dir}/skip_10_sec.mp3 -map 0:a:0 -map 1:a:0 {data_dir}/test_dual_stream.mp4"])
    return os.path.abspath(f'{data_dir}/test_dual_stream.mp4')

@pytest.fixture(scope='session')
def large_mp3_file(data_dir, small_mp3_file):
    print(f"Running: ffmpeg -f concat -safe 0 -i <(for i in {{1..50}} ; do echo 'file {small_mp3_file}'; done) {data_dir}/test_long.mp3")
    subprocess.call(["bash", "-c", f"ffmpeg -y -f concat -safe 0 -i <(for i in {{1..50}} ; do echo \"file {small_mp3_file}\"; done) -c copy {data_dir}/test_long.mp3"])
    return os.path.abspath(f'{data_dir}/test_long.mp3')

@pytest.fixture(scope='session')
def small_m4a_file(data_dir, small_mp3_file):
    file = os.path.abspath(f"{data_dir}/test.m4a")
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {small_mp3_file} {file}"])
    return file

@pytest.fixture(scope='session')
def small_ogg_file(data_dir, small_mp3_file):
    file = os.path.abspath(f"{data_dir}/test.ogg")
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {small_mp3_file} {file}"])
    return file

@pytest.fixture(scope='session')
def small_wav_file(data_dir, small_mp3_file):
    file = os.path.abspath(f"{data_dir}/test.wav")
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {small_mp3_file} {file}"])
    return file

@pytest.fixture(scope='session')
def small_wav_file_with_mp4_ext(data_dir, small_mp3_file):
    file = os.path.abspath(f"{data_dir}/test.wav")
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {small_mp3_file} {file}"])
    old_file = file
    file = file.replace(".mp4", ".wav")
    os.rename(old_file, file)
    return file

@pytest.fixture(scope='session')
def small_flac_file(data_dir, small_wav_file):
    file = os.path.abspath(f"{data_dir}/test.flac")
    subprocess.call(["bash", "-c", f"ffmpeg -y -i {small_wav_file} {file}"])
    return file

@pytest.fixture(scope='session')
def very_bad_mp3_file(data_dir, small_mp3_file):
    file = os.path.abspath(f"{data_dir}/very_bad_file.mp3")
    data = open(small_mp3_file, 'rb').read()
    #add some zeros in the middle.
    data = data[:len(data)//2] + b'\x00'*1000 + data[len(data)//2:]
    open(file, 'wb').write(data)
    return file

@pytest.fixture(scope='session')
def bad_meta_data_mp3(data_dir):
    file = os.path.abspath(f"{__file__}/../assets/audio_files/bad_meta_data.mp3")
    return file

@pytest.fixture(scope='session')
def non_zero_start_time_mp4(data_dir):
    file = os.path.abspath(f"{__file__}/../assets/audio_files/non_zero_start_time.mp4")
    return file

@pytest.fixture(scope='session')
def very_very_short_mp3_file(data_dir):
    file = os.path.abspath(f"{__file__}/../assets/audio_files/non_zero_start_time.mp4")
    return file

@pytest.fixture(scope='session')
def video_file_with_audio_in_stream_index_1(data_dir):
    file = os.path.abspath(f"{__file__}/../assets/audio_files/test.mov")
    return file

def test_24bitwav(small_24bit_wav_file):
    with AudioSample(small_24bit_wav_file) as au:
        nau = au.as_numpy()

def test_very_bad_mp3(very_bad_mp3_file):
    au = AudioSample(very_bad_mp3_file)
    nau = au[:].as_numpy()

def test_very_very_short_mp3(very_very_short_mp3_file):
    _ = AudioSample(very_very_short_mp3_file).as_numpy()

def test_dual_stream_file(dual_stream_mp4_file):
    with AudioSample(dual_stream_mp4_file) as au:
        assert au.streams == 2
    with AudioSample(dual_stream_mp4_file, stream_idx=0) as au:
        nau1 = au[0:3].as_numpy()
    with AudioSample(dual_stream_mp4_file, stream_idx=0) as au:
        nau1_10 = au[10.5:][:3].as_numpy()
    with AudioSample(dual_stream_mp4_file, stream_idx=1) as au:
        nau2 = au[0.5:][:3].as_numpy()
        sr = au.sample_rate
    au1 = AudioSample.from_numpy(nau1, rate=sr)
    memf = io.BytesIO()
    au1.write_to_file(memf)
    nau1_16 = AudioSample(memf, force_sample_rate=16000, force_channels=1).as_numpy()
    au1_10 = AudioSample.from_numpy(nau1_10, rate=sr)
    memf = io.BytesIO()
    au1_10.write_to_file(memf)
    nau1_10_16 = AudioSample(memf, force_sample_rate=16000, force_channels=1).as_numpy()
    au2 = AudioSample.from_numpy(nau2, rate=sr)
    memf = io.BytesIO()
    au2.write_to_file(memf)
    nau2_16 = AudioSample(memf, force_sample_rate=16000, force_channels=1).as_numpy()

    nau1_10_bak = nau1_10
    corr = correlate(nau1_10[0], nau2[0], mode='full')
    # Find the index of the maximum correlation
    max_index = np.argmax(corr)
    # Compute the time lag between the two waveforms
    time_lag = max_index - (nau1_10.shape[-1] - 1)
    if time_lag > 0:
        nau1_10 = nau1_10[:, time_lag:]
        nau2 = nau2[:,:-time_lag]
    if time_lag < 0:
        nau2 = nau2[:, time_lag:]
        nau1_10 = nau1_10[:, :-time_lag]

    assert np.all(np.isclose(nau1_10, nau2, atol=1e-1))


    nau1_10 = nau1_10_bak
    corr = correlate(nau1_10[0], nau1[0], mode='full')
    # Find the index of the maximum correlation
    max_index = np.argmax(corr)
    # Compute the time lag between the two waveforms
    time_lag = max_index - (nau1_10.shape[-1] - 1)
    if time_lag > 0:
        nau1_10 = nau1_10[:, time_lag:]
        nau1 = nau1[:,:-time_lag]
    if time_lag < 0:
        nau1 = nau1[:, time_lag:]
        nau1_10 = nau1_10[:, :-time_lag]

    assert not np.all(np.isclose(nau1_10, nau1, atol=1e-1))

def test_mem_file(small_mp3_file):
    memr = io.BytesIO()
    memw = io.BytesIO()
    memr.write(open(small_mp3_file, 'rb').read())
    memr.seek(0,0)
    with AudioSample(memr) as au:
        au.write_to_file(memw)
        assert AudioSample(memw).format == au.format    

def test_convert_file(data_dir, short_mp3_file, small_wav_file, small_opus_file):
    small_mp3_file = short_mp3_file
    with AudioSample(small_mp3_file) as au:
        au.write_to_file(f"{data_dir}/small_mp3_file-conv.m4a")
        with pytest.raises(ValueError, match="supported"):
            au.write_to_file(f"{data_dir}/small_mp3_file-conv.flac")
        au.write_to_file(f"{data_dir}/small_mp3_file-conv.ogg")
        with AudioSample(f"{data_dir}/small_mp3_file-conv.m4a") as au:
            assert au.format == 'mp4'
        with AudioSample(f"{data_dir}/small_mp3_file-conv.ogg") as au:
            assert au.format == 'ogg'

    with AudioSample(small_wav_file) as au:
        au.write_to_file(f"{data_dir}/small_mp3_file-conv.mp3", force_out_format='mp3')
        au.write_to_file(f"{data_dir}/small_mp3_file-conv.m4a", force_out_format='m4a')
        au.write_to_file(f"{data_dir}/small_mp3_file-conv.ogg", force_out_format='ogg')
        with AudioSample(f"{data_dir}/small_mp3_file-conv.m4a") as au:
            assert au.format == 'mp4'
        with AudioSample(f"{data_dir}/small_mp3_file-conv.ogg") as au:
            assert au.format == 'ogg'
        with AudioSample(f"{data_dir}/small_mp3_file-conv.mp3") as au:
            assert au.format == 'mp3'

    with AudioSample(small_opus_file) as au:
        au[0:1].write_to_file(f"{data_dir}/smaller-opus.opus")

    with AudioSample(f"{data_dir}/smaller-opus.opus") as au:
        assert au.format == 'ogg'

def test_video_mov(data_dir, video_file_with_audio_in_stream_index_1):
    with AudioSample(video_file_with_audio_in_stream_index_1) as au:
        au[0:1].write_to_file(f"{data_dir}/tmp-mov-file.mov", no_encode=True)
    with AudioSample(video_file_with_audio_in_stream_index_1) as au:
        au[0:0.5].write_to_file(f"{data_dir}/tmp-mov-file.mov", no_encode=False)


@pytest.mark.slow
def test_as_tensor(data_dir, small_mp3_file):
    au = AudioSample(open(small_mp3_file, "rb"))
    tau = au.as_tensor()

def test_short_samples(data_dir, small_mp3_file):
    au = AudioSample(open(small_mp3_file, "rb"))
    nau_cut = au[10:][:1].as_numpy()
    nau = au.as_numpy()
    #compare min, max, rms, encoder starts in a different position, results in slightly different outcome...
    assert nau[:, au.sample_rate*10:][:, :au.sample_rate*1].max() == nau_cut.max()
    assert nau[:, au.sample_rate*10:][:, :au.sample_rate*1].min() == nau_cut.min()
    assert np.isclose(np.sqrt((nau[:, au.sample_rate*10:][:, :au.sample_rate*1]**2).mean()), np.sqrt((nau_cut**2).mean()), atol=1e-3)
    with AudioSample(small_mp3_file) as au:
        au[10:][:1].write_sample_to_file(f"{data_dir}/test10-1.mp3", no_encode=True)

    with AudioSample(small_mp3_file, force_channels=1, thread_safe=True) as au:
        for i in range(int(au.duration)):
            au1sec = au[i:i+1]
            au1sec.f != au.f

    with AudioSample(small_mp3_file, force_channels=1, thread_safe=True) as au:
        assert au.format == 'mp3'

def test_various_file_types(small_m4a_file, small_ogg_file, small_wav_file, small_wav_file_with_mp4_ext, small_flac_file):
    with AudioSample(small_m4a_file) as au:
        assert au.format == 'mp4'
    with AudioSample(small_ogg_file) as au:
        assert au.format == 'ogg'
    with AudioSample(small_wav_file) as au:
        assert au.format == 'wav'
    with AudioSample(small_wav_file_with_mp4_ext) as au:
        assert au.format == 'wav'
    with AudioSample(small_flac_file) as au:
        assert au.format == 'flac'

    with AudioSample(small_ogg_file) as au:
        ogg_nau = au[10:11].as_numpy()
    with AudioSample(small_m4a_file) as au:
        m4a_nau = au[10:11].as_numpy()
    with AudioSample(small_flac_file) as au:
        flac_nau = au[10:11].as_numpy()
    with AudioSample(small_wav_file_with_mp4_ext) as au:
        wav_nau = au[10:11].as_numpy()

    assert np.isclose(np.sqrt((ogg_nau**2).mean()), np.sqrt((m4a_nau**2).mean()), atol=1e-2)
    assert np.all(np.isclose((flac_nau**2), (wav_nau**2), atol=1e-8))
    assert np.isclose(np.sqrt((ogg_nau**2).mean()), np.sqrt((wav_nau**2).mean()), atol=1e-2)

def test_long_samples(large_mp3_file, data_dir):
    t1 = time.time()
    with AudioSample(large_mp3_file) as au:
        print(f"{au.duration=} {au.sample_rate=} {au.format=}")
        au[100:][:1].write_sample_to_file(f"{data_dir}/test100-1.mp3", no_encode=True)
    #should take less than 100ms
    assert time.time() < t1+0.1
    t1 = time.time()
    with AudioSample(large_mp3_file, force_channels=1) as au:
        au[500:][:1].write_sample_to_file(f"{data_dir}/test10-1-s1.mp3", no_encode=True)
    #should take less than 100ms
    assert time.time() < t1+0.1
    t1 = time.time()
    with AudioSample(large_mp3_file, force_channels=1, thread_safe=True) as au:
        nau = au[500:][:1].as_numpy()
    #should take less than 1s
    assert time.time() < t1+0.1
    print(f"{(time.time() - t1)=}")

@pytest.mark.slow
def test_long_samples_slow(large_mp3_file, data_dir):
    with AudioSample(large_mp3_file, thread_safe=True) as au:
        split_3 = int(au.duration/3)
        for i in range(0, int(au.duration), split_3):
            au[i:i+split_3].write_to_file(f"{data_dir}/large_mp3_file-{i}.wav")
        au[0:split_3].write_to_file(f"{data_dir}/large_mp3_file-first.wav")
        subprocess.check_call(["diff", f"{data_dir}/large_mp3_file-0.wav", f"{data_dir}/large_mp3_file-first.wav"])

def test_bad_meta_data_read(bad_meta_data_mp3):
    with AudioSample(bad_meta_data_mp3) as au:
        nau = au.as_numpy()
    
def test_non_zero_start_time(non_zero_start_time_mp4):
    with AudioSample(non_zero_start_time_mp4) as au:
        nau = au.as_numpy()

    with AudioSample(non_zero_start_time_mp4) as au:
        nau = au[0:1].as_numpy()

def test_as_data(small_wav_file, short_mp3_file):
    small_mp3_file = short_mp3_file
    nau = AudioSample(small_wav_file).as_numpy()
    au = AudioSample.from_numpy(nau, rate=AudioSample(small_wav_file).sample_rate)
    nau2 = AudioSample(au.as_wav_data()).as_numpy()
    assert np.all(nau == nau2)
    nau2 = AudioSample(au.as_data()).as_numpy()
    assert np.all(nau == nau2)
    mp3_data = open(small_mp3_file,'rb').read()
    assert AudioSample(mp3_data).format == 'mp3'
    assert AudioSample(mp3_data).as_data()[0:3] == b'ID3'
    assert AudioSample(mp3_data).as_data(force_out_format='wav')[0:4] == b'RIFF'
    au_in = AudioSample(mp3_data)
    au_out = AudioSample(au_in.as_data())
    assert au_out.format == 'mp3'
    assert np.isclose(au_out.duration, au_in.duration, atol=1e-5)
    au_out = AudioSample(au_in.as_data(force_out_format='wav'))
    assert au_out.format == 'wav'
    assert np.isclose(au_out.duration, au_in.duration, atol=1e-1)

@pytest.mark.slow
def test_loop_on_big_file(large_mp3_file):
    au = AudioSample(large_mp3_file)
    for i in range(min(int(au.duration),600)):
        au[i:i+1].as_numpy()
    del au 
    au = AudioSample(large_mp3_file)
    for i in range(min(600, int(au.duration))-1, -1, -1):
        au[i:i+1].as_numpy()

def pool_fn(_au, t, x):
   _au[x:x+t].as_numpy()

def test_thread_pool(small_mp3_file):
    au = AudioSample(small_mp3_file, thread_safe=True)
    #test in multiple threads, wait for all to finish. this should not crash.
    with multiprocessing.pool.ThreadPool(4) as pool:
        pool.map(partial(pool_fn, au, int(au.duration/10)), range(0, int(au.duration), int(au.duration/10)))

def test_serializable(small_wav_file, small_mp3_file):
    au = AudioSample(small_wav_file)
    au2 = pickle.loads(pickle.dumps(au))
    au.as_wav_data() == au2.as_wav_data()
    au = AudioSample(small_mp3_file)
    au2 = pickle.loads(pickle.dumps(au))
    au.as_wav_data() == au2.as_wav_data()

def test_process_pool(small_mp3_file):
    au = AudioSample(small_mp3_file, thread_safe=True)
    #test in multiple threads, wait for all to finish. this should not crash.
    multiprocessing.set_start_method("spawn", force=True)
    with multiprocessing.Pool(4) as pool:
        pool.map(partial(pool_fn, au, int(au.duration/10)), range(0, int(au.duration), int(au.duration/10)))

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

def test_http_stream(small_mp3_file):
    #server small mp3 file in a thread
    au = AudioSample(small_mp3_file)
    au.as_numpy()
    import http.server
    import threading
    import socketserver
    PORT = 54321
    #allow address reuse
    os.chdir(os.path.dirname(small_mp3_file))
    Handler = http.server.SimpleHTTPRequestHandler
    Handler.extensions_map.update({
        ".mp3": "audio/mpeg",
    })
    #set default dir 
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("localhost", PORT), Handler)
    httpd_thread = threading.Thread(target=httpd.serve_forever)
    httpd_thread.daemon = True
    httpd_thread.start()
    time.sleep(0.1)
    au = AudioSample(f"http://localhost:{PORT}/{os.path.basename(small_mp3_file)}", thread_safe=True)
    nau = au.as_numpy()
    au = AudioSample(f"http://localhost:{PORT}/{os.path.basename(small_mp3_file)}", thread_safe=True)
    au2 = AudioSample(force_channels=2, thread_safe=True)
    au2 += au[0:1]
    nau = au2.as_numpy(mono_1d=False)
    httpd.shutdown()
    httpd.server_close()
    httpd_thread.join()

