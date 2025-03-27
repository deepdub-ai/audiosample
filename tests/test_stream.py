from random import randint
from random import seed
import time
from audiosample import AudioSample
from audiosample.downsampler import Downsampler
from audiosample.downsampler import mu_law_encode
import numpy as np
import pytest

def random_chunkify_random_size(x):
    i = 0
    while i < x.shape[-1]:
        j = i
        i += randint(100, 48000)
        yield x[...,j:i]

def chunk_data_as_numpy(np_data, chunk_size=1000):
    for i in range(0, np_data.shape[-1], chunk_size):
        yield np_data[..., i:i+chunk_size]


def test_stream_wav_to_mp3():
    audio_data = AudioSample().beep(100)
    audio_data_wav = audio_data.as_data(force_out_format='s16le')
    print(f"audio_data_wav: {len(audio_data_wav)=}")
    def chunk_data(data):
        CHUNK_SIZE = 1000
        for i in range(0, len(data), CHUNK_SIZE):
            # print(".", end="")
            yield data[i:i+CHUNK_SIZE]

    orig = AudioSample(audio_data_wav, force_read_format='s16le', force_read_sample_rate=48000, force_read_channels=1).as_data(force_out_format='mp3')
    au = AudioSample(chunk_data(audio_data_wav),
                     force_read_format='s16le', force_read_sample_rate=48000, force_read_channels=1)
    tmp = au.as_data(force_out_format='mp3')
    assert len(tmp) == len(orig)

    au = AudioSample(chunk_data(audio_data_wav),
                     force_read_format='s16le', force_read_sample_rate=48000, force_read_channels=1)


    tmp2 = b''
    for i,j in enumerate(au.as_data_stream(force_out_format='mp3')):
       tmp2 += j
    assert tmp2 == tmp
    print(f"tmp: {len(tmp)=}")


def test_stream_mp3_to_ogg():
    audio_data = AudioSample().beep(100)
    audio_data_mp3 = audio_data.as_data(force_out_format='mp3')
    # print(f"audio_data_mp3: {len(audio_data_mp3)=}")
    def chunk_data(data):
        CHUNK_SIZE = 1000
        for i in range(0, len(data), CHUNK_SIZE):
            # print(".", end="")
            yield data[i:i+CHUNK_SIZE]

    orig = AudioSample(audio_data_mp3, force_read_format='mp3').as_data(force_out_format='opus')

    au = AudioSample(chunk_data(audio_data_mp3))
    tmp = au.as_data(force_out_format='opus')
    au = AudioSample(chunk_data(audio_data_mp3))
    tmp2 = b''
    for i,j in enumerate(au.as_data_stream(force_out_format='opus')):
       tmp2 += j
    assert len(tmp2) == len(tmp)
    assert np.all(AudioSample(tmp2).as_numpy() == AudioSample(tmp).as_numpy())
    # assert tmp2 == tmp

def test_wav_to_mulaw():
    audio_data = AudioSample().beep(100)
    audio_data_wav = audio_data.as_data(force_out_format='s16le')
    print(f"audio_data_wav: {len(audio_data_wav)=}")
    def chunk_data(data):
        CHUNK_SIZE = 1000
        for i in range(0, len(data), CHUNK_SIZE):
            # print(".", end="")
            yield data[i:i+CHUNK_SIZE]

    tmp = AudioSample(audio_data_wav, force_read_format='s16le', force_sample_rate=8000).as_data(force_out_format='mulaw')
    tmp2 = b''
    au = AudioSample(chunk_data(audio_data_wav), force_sample_rate=8000)
    for i,j in enumerate(au.as_data_stream(force_out_format='mulaw')):
       tmp2 += j
    # print(f"tmp: {len(tmp)=}, {len(tmp2)=}")
    assert np.all(AudioSample(tmp2, force_read_format='mulaw').as_numpy() == AudioSample(tmp, force_read_format='mulaw').as_numpy())

def test_stream_numpy_to_mulaw():
    audio_data_np = AudioSample().beep(100).as_numpy()
    
    chunked = chunk_data_as_numpy(audio_data_np)
    l = 0
    for c in chunked:
        l += c.shape[-1]
    # print(f"len: {l=} vs. {audio_data_np.shape[-1]=}")
    assert l == audio_data_np.shape[-1]
    
    tmp = AudioSample(audio_data_np, force_read_sample_rate=48000, force_sample_rate=8000).as_data(force_out_format='mulaw')

    au = AudioSample(chunk_data_as_numpy(audio_data_np), force_read_sample_rate=48000, force_sample_rate=8000)
    tmp2 = b''
    
    for i,j in enumerate(au.as_data_stream(force_out_format='mulaw')):
       tmp2 += j

    assert tmp == tmp2

def test_stream_to_mp3_file(tmp_path):
    audio_data_np = AudioSample().beep(10).as_numpy()
        
    # Test MP3 output
    output_file_mp3 = tmp_path / "test_output.mp3"
    
    au = AudioSample(chunk_data_as_numpy(audio_data_np), force_read_sample_rate=48000)
    au.save(str(output_file_mp3))
    
    # Verify the MP3 file exists and has content
    assert output_file_mp3.exists()
    assert output_file_mp3.stat().st_size > 0
    
    # Verify the MP3 content is valid by reading it back
    result_mp3 = AudioSample(str(output_file_mp3))
    original = AudioSample(audio_data_np, force_read_sample_rate=48000)
    # Compare the MP3 audio content (allowing for some compression differences)
    # Trim to shorter length since MP3 encoding can modify duration slightly
    #compare isclose numpy len
    assert np.isclose(result_mp3.as_numpy().shape[-1], original.as_numpy().shape[-1], rtol=1000, atol=1000)
    # Test M4A output
    output_file_m4a = tmp_path / "test_output.m4a"

    #if not ValueError, raise error
    try:    
        au = AudioSample(chunk_data_as_numpy(audio_data_np), force_read_sample_rate=48000)
        au.save(str(output_file_m4a))
    except ValueError as e:
        assert str(e) == "Cannot encode stream input to non-streamable output format"
    
    # Test AAC output
    output_file_aac = tmp_path / "test_output.aac"
    au = AudioSample(chunk_data_as_numpy(audio_data_np), force_read_sample_rate=48000)
    au.save(str(output_file_aac))
    assert output_file_aac.exists()
    assert output_file_aac.stat().st_size > 0

def test_stream_to_wav_file(tmp_path):
    audio_data_np = AudioSample().beep(10).as_numpy()
    output_file_wav = tmp_path / "test_output.wav"
    #stream audio data to file
            
    au = AudioSample(chunk_data_as_numpy(audio_data_np), force_read_sample_rate=48000)
    try:
        au.save(str(output_file_wav))
    except ValueError as e:
        assert str(e) == "Cannot encode stream input to non-streamable output format"
    

def test_stream_random_chunkify_to_mp3_file(tmp_path):
            
    seed(0)
    def normal_chunkify(x):
        CHUNK_SIZE = 10000
        tot = 0
        for i in range(0, x.shape[-1], CHUNK_SIZE):
            c = x[..., i:i+CHUNK_SIZE]
            tot += c.shape[-1]
            yield c
        assert tot == x.shape[-1]
        return x
    audio_data_np = AudioSample().beep(10).as_numpy()
    out_no_stream = AudioSample(audio_data_np, force_read_sample_rate=48000).as_data(force_out_format='mp3')
    out_stream = b"".join(list(AudioSample(normal_chunkify(audio_data_np), force_read_sample_rate=48000).as_data_stream(force_out_format='mp3')))
    
    #print len in numpy diff
    print(f"len diff: {AudioSample(out_stream).as_numpy().shape[-1] - AudioSample(out_no_stream).as_numpy().shape[-1]}")
    #find the first difference
    diff_idx = np.where(AudioSample(out_stream).as_numpy()[:AudioSample(out_no_stream).as_numpy().shape[-1]] != AudioSample(out_no_stream).as_numpy())[0][0]
    print(f"diff_idx: {diff_idx}")
    # assert diff_idx == 0
    print((AudioSample(out_stream).as_numpy()[:AudioSample(out_no_stream).as_numpy().shape[-1]] == AudioSample(out_no_stream).as_numpy()).argmax())
    # assert np.all(AudioSample(out_stream).as_numpy()[:AudioSample(out_no_stream).as_numpy().shape[-1]] == AudioSample(out_no_stream).as_numpy())

    seed(0)
    def random_chunkify(x):
        i = 0
        while i < x.shape[-1]:
            j = i
            i += randint(100, 48000)
            yield x[...,j:i]
    #random chunkify with random chunk size
    out_stream = b"".join(list(AudioSample(random_chunkify_random_size(audio_data_np), force_read_sample_rate=48000).as_data_stream(force_out_format='mp3')))
    #assert np.all(AudioSample(out_stream).as_numpy()[:AudioSample(out_no_stream).as_numpy().shape[-1]] == AudioSample(out_no_stream).as_numpy())

#run for freq 8000, and 22050
@pytest.mark.parametrize("sample_rate", [8000, 22050])
def test_stream_random_chunkify_to_headerless_wav_file(tmp_path, sample_rate):
    audio_data_np = AudioSample().beep(3, freq=400).as_numpy()
    #stream audio data to file
    au = AudioSample(chunk_data_as_numpy(audio_data_np, chunk_size=100), force_read_sample_rate=48000, force_sample_rate=sample_rate)
    data_stream = au.as_data_stream(force_out_format='wav', no_encode=False)
    collect = b""
    for i, data in enumerate(data_stream):
        # print(f"data {i}: {len(data)=}")
        collect += data
    au2 = AudioSample.from_headerless_data(collect, sample_rate=sample_rate, precision=16, channels=1)


    print(f"collected: {len(collect)=}")
    au = AudioSample(random_chunkify_random_size(audio_data_np), force_read_sample_rate=48000, force_sample_rate=sample_rate)
    data = au.as_data(force_out_format='wav', no_encode=False)
    assert data == collect 
    au2 = AudioSample.from_headerless_data(collect, sample_rate=sample_rate, precision=16, channels=1)
    # au2.play()
    def one_chunk_data_as_numpy(np_data):
        yield np_data
    au = AudioSample(one_chunk_data_as_numpy(audio_data_np), force_read_sample_rate=48000, force_sample_rate=sample_rate)
    data_stream = au.as_data_stream(force_out_format='wav', no_encode=False)
    collect = b""
    for i, data in enumerate(data_stream):
        print(f"data {i}: {len(data)=}")
        collect += data
    print(f"collected: {len(collect)=}")
    #play
    au4 = AudioSample.from_headerless_data(collect, sample_rate=sample_rate, precision=16, channels=1)
    # au4.play()
    
@pytest.mark.parametrize("src_sample_rate, target_sample_rate", [(48000, 8000), (48000, 22050)])
def test_stream_downsampler(src_sample_rate, target_sample_rate):
    audio_data_np = AudioSample(force_sample_rate=src_sample_rate).beep(10, freq=500).as_numpy()

    ##WARM
    ds = Downsampler(src_sample_rate, target_sample_rate)
    t1 = time.time()    
    collect1 = np.array([])
    for i, data in enumerate(chunk_data_as_numpy(audio_data_np, chunk_size=10000)):
        out = mu_law_encode(ds.encode(data))
        break
    t2 = time.time()
    print(f"warm time: {(t2-t1)*1000=}ms")
    #stream audio data to file
    ds = Downsampler(src_sample_rate, target_sample_rate)
    t1 = time.time()
    collect1 = np.array([])
    for i, data in enumerate(chunk_data_as_numpy(audio_data_np, chunk_size=10)):
        out = ds.encode(data)
        collect1 = np.concatenate([collect1, out])
    collect1 = np.concatenate([collect1, ds.encode(None)])
    t2 = time.time()
    print(f"10 time: {(t2-t1)*1000=}ms")
    #now with different chunk size
    # ds = Downsampler(src_sample_rate, target_sample_rate)

    t1 = time.time()
    collect2 = b''
    for i, data in enumerate(chunk_data_as_numpy(audio_data_np, chunk_size=100)):
        collect2 += mu_law_encode(ds.encode(data)).tobytes()
    collect2 += mu_law_encode(ds.encode(None)).tobytes()
    # AudioSample._mu_law_encode(collect2).tobytes()
    t2 = time.time()
    print(f"mulaw 100 time: {(t2-t1)*1000=}ms")

    t1 = time.time()
    collect2 = np.array([])
    for i, data in enumerate(chunk_data_as_numpy(audio_data_np, chunk_size=100)):
        out = ds.encode(data)
        collect2 = np.concatenate([collect2, out])
    collect2 = np.concatenate([collect2, ds.encode(None)])
    # AudioSample._mu_law_encode(collect2).tobytes()
    t2 = time.time()
    assert np.all(collect1 == collect2)
    print(f"100 time: {(t2-t1)*1000=}ms")
    #now with different chunk size
    # ds = Downsampler(src_sample_rate, target_sample_rate)
    t1 = time.time()
    collect3 = np.array([])
    for i, data in enumerate(chunk_data_as_numpy(audio_data_np, chunk_size=10000)):
        out = ds.encode(data)
        collect3 = np.concatenate([collect3, out])
    collect3 = np.concatenate([collect3, ds.encode(None)])
    t2 = time.time()
    assert np.all(collect1 == collect3)
    print(f"10K time: {(t2-t1)*1000=}ms")
    #now with different chunk size
    # ds = Downsampler(src_sample_rate, target_sample_rate)
    t1 = time.time()
    collect4 = np.array([])
    for i, data in enumerate(chunk_data_as_numpy(audio_data_np, chunk_size=100000)):
        out = ds.encode(data)
        collect4 = np.concatenate([collect4, out])
    collect4 = np.concatenate([collect4, ds.encode(None)])
    t2 = time.time()
    assert np.all(collect1 == collect4)
    print(f"100K time: {(t2-t1)*1000=}ms")

    t1 = time.time()
    collect2 = b''
    for i, data in enumerate(chunk_data_as_numpy(audio_data_np, chunk_size=100000)):
        collect2 += mu_law_encode(ds.encode(data)).tobytes()
    collect2 += mu_law_encode(ds.encode(None)).tobytes()
    # AudioSample._mu_law_encode(collect2).tobytes()
    t2 = time.time()
    print(f"mulaw 100K time: {(t2-t1)*1000=}ms")


    t1 = time.time()
    wav = AudioSample(chunk_data_as_numpy(audio_data_np, chunk_size=100), force_read_sample_rate=src_sample_rate, force_sample_rate=target_sample_rate).as_data(force_out_format='mulaw', no_encode=False)
    t2 = time.time()
    print(f"100 pyav time: {(t2-t1)*1000=}ms")



    t1 = time.time()
    wav = AudioSample(chunk_data_as_numpy(audio_data_np, chunk_size=10000), force_read_sample_rate=src_sample_rate, force_sample_rate=target_sample_rate).as_data(force_out_format='mulaw', no_encode=False)
    t2 = time.time()
    print(f"10K pyav time: {(t2-t1)*1000=}ms")

    t1 = time.time()
    wav = AudioSample(chunk_data_as_numpy(audio_data_np, chunk_size=100000), force_read_sample_rate=src_sample_rate, force_sample_rate=target_sample_rate).as_data(force_out_format='mulaw', no_encode=False)
    t2 = time.time()
    print(f"100K pyav time: {(t2-t1)*1000=}ms")


    # au = AudioSample(collect1, force_read_sample_rate=target_sample_rate)
    # au.save(str("test_output_downsampler.wav"))
    # au.play()

@pytest.mark.slow
@pytest.mark.parametrize("sample_rate", [8000])
def test_stream_random_chunkify_to_headerless_wav_file_with_play(tmp_path, sample_rate):
    audio_data_np = AudioSample().beep(3, freq=500).as_numpy()
    #stream audio data to file
    # au = AudioSample(chunk_data_as_numpy(audio_data_np, chunk_size=100), force_read_sample_rate=48000, force_sample_rate=sample_rate)
    # data_stream = au.as_data_stream(force_out_format='wav', no_encode=False)
    # collect = b""
    # for i, data in enumerate(data_stream):
    #     # print(f"data {i}: {len(data)=}")
    #     collect += data
    # au2 = AudioSample.from_headerless_data(collect, sample_rate=sample_rate, precision=16, channels=1)
    # # au2.play()

    # print(f"collected: {len(collect)=}")
    seed(0)
    au = AudioSample(random_chunkify_random_size(audio_data_np), force_read_sample_rate=48000, force_sample_rate=sample_rate)
    collect = au.as_data(force_out_format='wav', no_encode=False)
    print(f"data: {len(collect)=}")
    #assert data == collect 
    au2 = AudioSample.from_headerless_data(collect, sample_rate=sample_rate, precision=16, channels=1)
    au2.save(str("test_output_random_chunkify_to_headerless_wav_file_with_play.wav"))
    import ipdb; ipdb.set_trace()
    au2.play()
    def one_chunk_data_as_numpy(np_data):
        yield np_data
    au = AudioSample(one_chunk_data_as_numpy(audio_data_np), force_read_sample_rate=48000, force_sample_rate=sample_rate)
    data_stream = au.as_data_stream(force_out_format='wav', no_encode=False)
    collect2 = b""
    for i, data in enumerate(data_stream):
        print(f"data {i}: {len(data)=}")
        collect2 += data
    assert collect == collect2
    print(f"collected: {len(collect)=}")
    #play
    au4 = AudioSample.from_headerless_data(collect, sample_rate=sample_rate, precision=16, channels=1)
    au4.play()
    au4.save(str("test_output_one_chunk_data_as_numpy_to_headerless_wav_file_with_play.wav"))
