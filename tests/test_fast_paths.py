"""Tests for the audiosample_rs native fast path."""
import math
import os
import struct

import pytest

from audiosample import AudioSample
from audiosample import fast_paths

rs = pytest.importorskip("audiosample_rs", reason="audiosample_rs not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_s16le(n: int, sr: int = 48_000, freq: float = 400.0) -> bytes:
    """Return `n` samples of a 400 Hz sine wave as s16le bytes."""
    return struct.pack(
        "<" + "h" * n,
        *[int(16000 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)],
    )


def _make_audiosample(raw: bytes, in_sr: int, out_sr: int) -> AudioSample:
    return AudioSample(
        raw,
        force_read_format="s16le",
        force_read_sample_rate=in_sr,
        force_sample_rate=out_sr,
    )


# ---------------------------------------------------------------------------
# is_supported / can_handle
# ---------------------------------------------------------------------------

class TestIsSupported:
    def test_mulaw_48_to_16(self):
        assert rs.is_supported("s16le", 48_000, "mulaw", 16_000, 1)

    def test_mulaw_48_to_8(self):
        assert rs.is_supported("s16le", 48_000, "pcm_mulaw", 8_000, 1)

    def test_s16le_48_to_16(self):
        assert rs.is_supported("s16le", 48_000, "s16le", 16_000, 1)

    def test_s16le_48_to_8(self):
        assert rs.is_supported("s16le", 48_000, "pcm_s16le", 8_000, 1)

    def test_unsupported_44100(self):
        assert not rs.is_supported("s16le", 44_100, "mulaw", 16_000, 1)

    def test_unsupported_stereo(self):
        assert not rs.is_supported("s16le", 48_000, "mulaw", 16_000, 2)

    def test_unsupported_out_format(self):
        assert not rs.is_supported("s16le", 48_000, "mp3", 16_000, 1)


class TestCanHandle:
    def test_explicit_force_read_format(self):
        raw = _sine_s16le(4800)
        a = _make_audiosample(raw, 48_000, 16_000)
        assert fast_paths.can_handle(a, "mulaw")
        assert fast_paths.can_handle(a, "s16le")

    def test_wave_header_inference(self):
        """from_headerless_data doesn't set force_read_format — infer from wave_header."""
        raw = _sine_s16le(4800)
        inner = AudioSample.from_headerless_data(raw, sample_rate=48_000)
        a = AudioSample(inner, force_sample_rate=16_000)
        assert fast_paths._read_format_for(a) == "s16le"
        assert fast_paths._input_sample_rate(a) == 48_000
        assert fast_paths.can_handle(a, "s16le")

    def test_disabled_env_var(self, monkeypatch):
        # Patch _DISABLED directly — reloading the module would leave _DISABLED=True
        # for all subsequent tests since monkeypatch only restores the env var.
        monkeypatch.setattr(fast_paths, "_DISABLED", True)
        raw = _sine_s16le(4800)
        a = _make_audiosample(raw, 48_000, 16_000)
        assert not fast_paths.can_handle(a, "mulaw")


# ---------------------------------------------------------------------------
# mulaw output
# ---------------------------------------------------------------------------

class TestMulawOutput:
    @pytest.mark.parametrize("out_sr,factor", [(16_000, 3), (8_000, 6)])
    def test_output_size(self, out_sr, factor):
        n = 48_000
        raw = _sine_s16le(n)
        a = _make_audiosample(raw, 48_000, out_sr)
        out = a.as_data(force_out_format="mulaw")
        assert len(out) == n // factor

    @pytest.mark.parametrize("out_sr", [16_000, 8_000])
    def test_streaming_equals_batch(self, out_sr):
        raw = _sine_s16le(48_000)
        batch = _make_audiosample(raw, 48_000, out_sr).as_data(force_out_format="mulaw")
        stream = b"".join(
            _make_audiosample(raw, 48_000, out_sr).as_data_stream(force_out_format="mulaw")
        )
        assert batch == stream


# ---------------------------------------------------------------------------
# s16le output
# ---------------------------------------------------------------------------

class TestS16leOutput:
    @pytest.mark.parametrize("out_sr,factor", [(16_000, 3), (8_000, 6)])
    def test_output_size(self, out_sr, factor):
        n = 48_000
        raw = _sine_s16le(n)
        a = _make_audiosample(raw, 48_000, out_sr)
        out = a.as_data(force_out_format="s16le")
        assert len(out) == (n // factor) * 2  # 2 bytes per sample

    @pytest.mark.parametrize("out_sr", [16_000, 8_000])
    def test_streaming_equals_batch(self, out_sr):
        raw = _sine_s16le(48_000)
        batch = _make_audiosample(raw, 48_000, out_sr).as_data(force_out_format="s16le")
        stream = b"".join(
            _make_audiosample(raw, 48_000, out_sr).as_data_stream(force_out_format="s16le")
        )
        assert batch == stream

    def test_samples_are_valid_i16(self):
        raw = _sine_s16le(4800)
        a = _make_audiosample(raw, 48_000, 16_000)
        out = a.as_data(force_out_format="s16le")
        samples = struct.unpack("<" + "h" * (len(out) // 2), out)
        assert all(-32768 <= s <= 32767 for s in samples)


# ---------------------------------------------------------------------------
# from_headerless_data path
# ---------------------------------------------------------------------------

class TestFromHeaderlessData:
    def test_force_sample_rate_applied(self):
        """AudioSample(inner_as, force_sample_rate=X) must not silently drop X."""
        raw = _sine_s16le(4800)
        inner = AudioSample.from_headerless_data(raw, sample_rate=48_000)
        a = AudioSample(inner, force_sample_rate=16_000)
        assert a.force_sample_rate == 16_000

    def test_routes_to_rust(self):
        raw = _sine_s16le(4800)
        a = AudioSample(
            AudioSample.from_headerless_data(raw, sample_rate=48_000),
            force_sample_rate=16_000,
        )
        assert fast_paths.can_handle(a, "s16le")
        assert fast_paths.can_handle(a, "mulaw")

    def test_correct_output_size_mulaw(self):
        n = 48_000
        raw = _sine_s16le(n)
        out = AudioSample(
            AudioSample.from_headerless_data(raw, sample_rate=48_000),
            force_sample_rate=16_000,
        ).as_data(force_out_format="mulaw")
        assert len(out) == n // 3

    def test_correct_output_size_s16le(self):
        n = 48_000
        raw = _sine_s16le(n)
        out = AudioSample(
            AudioSample.from_headerless_data(raw, sample_rate=48_000),
            force_sample_rate=8_000,
        ).as_data(force_out_format="s16le")
        assert len(out) == (n // 6) * 2

    def test_matches_explicit_path(self):
        """from_headerless_data path must produce identical bytes to the explicit path."""
        n = 48_000
        raw = _sine_s16le(n)
        explicit = _make_audiosample(raw, 48_000, 16_000).as_data(force_out_format="mulaw")
        headerless = AudioSample(
            AudioSample.from_headerless_data(raw, sample_rate=48_000),
            force_sample_rate=16_000,
        ).as_data(force_out_format="mulaw")
        assert explicit == headerless
