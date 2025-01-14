from random import randint
from random import seed
from audiosample import AudioSample
import numpy as np
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
    def chunk_data_as_numpy(np_data):
        CHUNK_SIZE = 1000
        for i in range(0, np_data.shape[-1], CHUNK_SIZE):
            # print(".", end="")
            yield np_data[..., i:i+CHUNK_SIZE]

    
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
    
    def chunk_data_as_numpy(np_data):
        CHUNK_SIZE = 1000
        for i in range(0, np_data.shape[-1], CHUNK_SIZE):
            yield np_data[..., i:i+CHUNK_SIZE]
    
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
    def chunk_data_as_numpy(np_data):
        CHUNK_SIZE = 1000
        for i in range(0, np_data.shape[-1], CHUNK_SIZE):
            yield np_data[..., i:i+CHUNK_SIZE]
            
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
    def random_chunkify_random_size(x):
        i = 0
        while i < x.shape[-1]:
            j = i
            i += randint(100, 48000)
            yield x[...,j:i]
    out_stream = b"".join(list(AudioSample(random_chunkify_random_size(audio_data_np), force_read_sample_rate=48000).as_data_stream(force_out_format='mp3')))
    #assert np.all(AudioSample(out_stream).as_numpy()[:AudioSample(out_no_stream).as_numpy().shape[-1]] == AudioSample(out_no_stream).as_numpy())
