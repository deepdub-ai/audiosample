"""Optional native fast paths for AudioSample.

If the companion package ``audiosample_rs`` is installed and the requested
input/output configuration is supported, ``as_data_stream`` will dispatch
to it instead of going through PyAV. Otherwise this module is a no-op and
audiosample behaves exactly as before.

Set the env var ``AUDIOSAMPLE_DISABLE_RS=1`` to force the slow path
unconditionally (useful for A/B testing or working around bugs).
"""
from __future__ import annotations

import io
import os
from typing import Generator, Iterable, Optional

try:
    import audiosample_rs as _rs  # type: ignore
    HAS_RS = True
except ImportError:
    _rs = None  # type: ignore
    HAS_RS = False

_DISABLED = os.environ.get("AUDIOSAMPLE_DISABLE_RS", "0") not in ("0", "", "false", "False")


def is_available() -> bool:
    """Return True if the native fast path is importable AND not disabled by env."""
    return HAS_RS and not _DISABLED


def _read_format_for(self) -> Optional[str]:
    """Return the s16le/etc input format string if known, else None.

    Checks force_read_format first; falls back to inferring from wave_header
    (type_of_format=1 is PCM, precision gives the bit depth).
    """
    fmt = getattr(self, "force_read_format", None)
    if fmt:
        return fmt
    wh = getattr(self, "wave_header", None)
    if wh and getattr(wh, "type_of_format", None) == 1:
        return {16: "s16le", 24: "s24le", 32: "s32le"}.get(getattr(wh, "precision", 0))
    return None


def _channels_for(self) -> int:
    return int(getattr(self, "channels", 1) or 1)


def _input_sample_rate(self) -> Optional[int]:
    if getattr(self, "force_read_sample_rate", None):
        return int(self.force_read_sample_rate)
    wh = getattr(self, "wave_header", None)
    sr = wh and getattr(wh, "sample_rate", None)
    return int(sr) if sr else None


def can_handle(self, force_out_format: Optional[str]) -> bool:
    """Return True if the configured AudioSample + requested output can be done in Rust."""
    if not is_available():
        return False
    if not force_out_format:
        return False
    in_fmt = _read_format_for(self)
    in_sr = _input_sample_rate(self)
    out_sr = int(getattr(self, "force_sample_rate", None) or 0)
    ch = _channels_for(self)
    if not in_fmt or not in_sr or not out_sr:
        return False
    return bool(_rs.is_supported(in_fmt, in_sr, force_out_format, out_sr, ch))


def stream(self, force_out_format: str) -> Generator[bytes, None, None]:
    """Yield converted bytes using the native resampler.

    Handles two input shapes:
      - iterable_input_buffer: a generator of raw bytes chunks (streaming).
      - in-memory bytes (self._data or self.f as BytesIO).
    """
    assert is_available()
    in_fmt = _read_format_for(self)
    in_sr = _input_sample_rate(self)
    out_sr = int(self.force_sample_rate)
    ch = _channels_for(self)
    resampler = _rs.Resampler(in_fmt, in_sr, force_out_format, out_sr, ch)

    iterable = getattr(self, "iterable_input_buffer", None)
    if iterable is not None:
        # Hold over the trailing odd byte across feeds (s16le requires even
        # byte counts).
        carry = bytearray()
        first = _drain_already_buffered(self)
        if first:
            for chunk in _feed_with_carry(resampler, first, carry):
                yield chunk
        for raw in iterable:
            buf = _coerce_chunk_to_bytes(raw, ch)
            if not buf:
                continue
            for chunk in _feed_with_carry(resampler, buf, carry):
                yield chunk
        # Any final carry byte is dropped: it's an unmatched half-sample.
        tail = resampler.flush()
        if tail:
            yield tail
        return

    # Batch case: data already in memory.
    data = getattr(self, "_data", None) or b""
    if not data:
        f = getattr(self, "f", None)
        if isinstance(f, io.BytesIO):
            data = f.getvalue()
    if not data:
        raise RuntimeError("audiosample fast path: no input data available")
    chunk = resampler.feed(bytes(data))
    if chunk:
        yield chunk
    tail = resampler.flush()
    if tail:
        yield tail


def _feed_with_carry(resampler, buf: bytes, carry: bytearray) -> Iterable[bytes]:
    """Feed `buf` to the resampler, prefixing any held-over odd byte and
    holding back a new odd byte if `buf`'s combined length is odd."""
    if carry:
        buf = bytes(carry) + buf
        carry.clear()
    if len(buf) & 1:
        carry.extend(buf[-1:])
        buf = buf[:-1]
    if not buf:
        return
    out = resampler.feed(buf)
    if out:
        yield out


def _drain_already_buffered(self) -> bytes:
    """If the AudioSample pulled some bytes off the iterable into self.f to
    sniff the format, return them and clear the buffer. Otherwise empty."""
    f = getattr(self, "f", None)
    if isinstance(f, io.BytesIO):
        buf = f.getvalue()
        f.seek(0)
        f.truncate(0)
        return buf
    return b""


def _coerce_chunk_to_bytes(chunk, channels: int) -> bytes:
    """Coerce a streaming chunk into raw s16le bytes."""
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        return bytes(chunk)
    # numpy array fallback
    try:
        import numpy as np  # local import to keep this file lightweight
    except ImportError:
        np = None  # type: ignore
    if np is not None and isinstance(chunk, np.ndarray):
        return chunk.astype("<i2").tobytes()
    raise TypeError(f"unsupported streaming chunk type {type(chunk)!r}")
