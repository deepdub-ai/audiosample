//! Native acceleration for audiosample.
//!
//! Optimised hot path (s16le mono 48 kHz → G.711 mu-law or s16le at 16/8 kHz):
//!   1. 64 KB mu-law LUT  – one table-lookup per output sample.
//!   2. Symmetric FIR fold – Hamming-windowed sinc is symmetric, so
//!      halve the multiplications: acc += h[k] * (x[i] + x[j]).
//!   3. Pre-convert i16→f32 once – avoid redundant casts in the inner loop.
//!   4. Monomorphise on (N_TAPS, FACTOR, MU) – compiler unrolls inner loop,
//!      eliminates branches, enables auto-SIMD.
//!   5. Explicit NEON (aarch64) / SSE (x86_64) – vectorise the dot product.

#[cfg(feature = "python")]
use pyo3::exceptions::{PyTypeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;

use std::sync::OnceLock;

pub mod flac;

// ============================================================================
// Output format
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
enum OutFmt { MuLaw, S16le }

impl OutFmt {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "mulaw" | "pcm_mulaw" => Some(OutFmt::MuLaw),
            "s16le" | "pcm_s16le" => Some(OutFmt::S16le),
            _ => None,
        }
    }
}

#[inline(always)]
fn push_sample(acc: f32, out: &mut Vec<u8>, lut: &[u8; 65536], out_fmt: OutFmt) {
    match out_fmt {
        OutFmt::MuLaw => out.push(unsafe { *lut.get_unchecked(clamp_i16(acc) as u16 as usize) }),
        OutFmt::S16le => out.extend_from_slice(&clamp_i16(acc).to_le_bytes()),
    }
}

// ============================================================================
// (1) 64 KB mu-law LUT
// ============================================================================

static MULAW_LUT: OnceLock<Box<[u8; 65536]>> = OnceLock::new();

/// Returns the global 64 KB mu-law lookup table, building it on first call.
pub fn mulaw_lut() -> &'static [u8; 65536] {
    MULAW_LUT.get_or_init(|| {
        let mut lut = Box::new([0u8; 65536]);
        for i in 0..65536u32 {
            lut[i as usize] = mulaw_encode_canonical(i as u16 as i16);
        }
        lut
    })
}

/// Canonical G.711 mu-law encoder — used to populate the LUT.
const fn mulaw_encode_canonical(pcm: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32635;
    let mut s = pcm as i32;
    let sign: u8 = if s < 0 { s = -s; 0x00 } else { 0x80 };
    if s > CLIP { s = CLIP; }
    s += BIAS;
    let v = ((s >> 7) & 0xFF) as u32;
    let exp: u8 = if v & 0x80 != 0 { 7 } else if v & 0x40 != 0 { 6 }
        else if v & 0x20 != 0 { 5 } else if v & 0x10 != 0 { 4 }
        else if v & 0x08 != 0 { 3 } else if v & 0x04 != 0 { 2 }
        else if v & 0x02 != 0 { 1 } else { 0 };
    let mantissa: u8 = ((s >> (exp as i32 + 3)) & 0x0F) as u8;
    !(sign | (exp << 4) | mantissa)
}

/// Encode one s16 sample to mu-law via the prebuilt LUT.
#[inline(always)]
pub fn linear_to_mulaw(pcm: i16) -> u8 {
    // SAFETY: index is u16 cast, always 0..65535.
    unsafe { *mulaw_lut().get_unchecked(pcm as u16 as usize) }
}

// ============================================================================
// FIR design
// ============================================================================

pub fn design_lowpass_fir(num_taps: usize, cutoff_norm: f32) -> Vec<f32> {
    assert!(num_taps % 2 == 1, "use odd taps for linear-phase symmetry");
    let m = (num_taps - 1) as f32;
    let two_pi = std::f32::consts::TAU;
    let pi = std::f32::consts::PI;
    let mut taps = vec![0.0f32; num_taps];
    let mut sum = 0.0f32;
    for n in 0..num_taps {
        let k = n as f32 - m * 0.5;
        let sinc = if k == 0.0 { 2.0 * cutoff_norm }
                   else { (two_pi * cutoff_norm * k).sin() / (pi * k) };
        let w = 0.54 - 0.46 * (two_pi * n as f32 / m).cos();
        taps[n] = sinc * w;
        sum += taps[n];
    }
    for t in taps.iter_mut() { *t /= sum; }
    taps
}

/// (2) Fold a symmetric FIR: return only the first half+centre (length N/2+1).
/// Since h[k] = h[N-1-k], the full dot product collapses to
///   Σ h[k] * (x[i] + x[j])  for k in 0..N/2,  plus h[N/2]*x[centre].
fn fold_taps(taps: &[f32]) -> Vec<f32> {
    taps[..=taps.len() / 2].to_vec()
}

#[inline(always)]
pub fn clamp_i16(x: f32) -> i16 {
    if x >= 32767.0 { 32767 } else if x <= -32768.0 { -32768 } else { x as i16 }
}

// ============================================================================
// (4+5) Dot-product kernels – monomorphised + arch-specialised
// ============================================================================

/// Symmetric dot product: win must have length N, ht must have length N/2+1.
/// Dispatches to NEON (aarch64), SSE (x86_64), or portable scalar.
#[inline(always)]
fn dot_sym<const N: usize>(win: &[f32], ht: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is always present on aarch64-linux.
    return unsafe { dot_neon::<N>(win, ht) };

    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is guaranteed on x86_64.
    return unsafe { dot_sse::<N>(win, ht) };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    dot_scalar::<N>(win, ht)
}

/// Scalar fallback: 4-independent-accumulator unroll for ILP.
/// LLVM will auto-vectorise this loop when target-cpu=native is set.
#[inline(always)]
#[allow(unused)]
fn dot_scalar<const N: usize>(win: &[f32], ht: &[f32]) -> f32 {
    let half = N / 2;
    let mut a0 = 0.0f32; let mut a1 = 0.0f32;
    let mut a2 = 0.0f32; let mut a3 = 0.0f32;
    let chunks4 = (half / 4) * 4;
    let mut k = 0usize;
    // SAFETY: k+3 < half <= N/2, so all indices are in-bounds.
    unsafe {
        while k < chunks4 {
            a0 += *ht.get_unchecked(k)   * (*win.get_unchecked(k)   + *win.get_unchecked(N-1-k));
            a1 += *ht.get_unchecked(k+1) * (*win.get_unchecked(k+1) + *win.get_unchecked(N-2-k));
            a2 += *ht.get_unchecked(k+2) * (*win.get_unchecked(k+2) + *win.get_unchecked(N-3-k));
            a3 += *ht.get_unchecked(k+3) * (*win.get_unchecked(k+3) + *win.get_unchecked(N-4-k));
            k += 4;
        }
        let mut acc = a0 + a1 + a2 + a3;
        while k < half {
            acc += *ht.get_unchecked(k) * (*win.get_unchecked(k) + *win.get_unchecked(N-1-k));
            k += 1;
        }
        acc + *ht.get_unchecked(half) * *win.get_unchecked(half)
    }
}

/// Explicit NEON kernel (aarch64).
/// Processes 4 symmetric pairs per iteration via vfmaq_f32.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_neon<const N: usize>(win: &[f32], ht: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let half = N / 2;
    let wp = win.as_ptr();
    let tp = ht.as_ptr();
    let mut acc = vdupq_n_f32(0.0f32);
    let full4 = half / 4;
    for i in 0..full4 {
        let k = i * 4;
        // Left side: win[k..k+4] — forward load.
        let left = vld1q_f32(wp.add(k));
        // Right side: win[N-1-k], win[N-2-k], win[N-3-k], win[N-4-k].
        // Load win[N-4-k..N-k] (forward) then reverse to [N-1-k, N-2-k, N-3-k, N-4-k].
        let raw   = vld1q_f32(wp.add(N - 1 - k - 3));
        // vrev64q reverses within each 64-bit lane:
        //   [a,b,c,d] → [b,a,d,c]
        let rev64 = vrev64q_f32(raw);
        // vextq_f32(v,v,2) rotates 2 lanes:
        //   [b,a,d,c] → [d,c,b,a]  (full 4-element reversal)
        let right = vextq_f32(rev64, rev64, 2);
        let pair  = vaddq_f32(left, right);
        let taps  = vld1q_f32(tp.add(k));
        acc = vfmaq_f32(acc, taps, pair);
    }
    // Horizontal sum of the 4-lane accumulator.
    let mut total = vaddvq_f32(acc);
    // Cleanup: remaining pairs (< 4).
    for k in full4 * 4..half {
        total += *tp.add(k) * (*wp.add(k) + *wp.add(N - 1 - k));
    }
    // Centre tap.
    total += *tp.add(half) * *wp.add(half);
    total
}

/// Explicit SSE kernel (x86_64). SSE2 is always available on x86_64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2,sse3")]
#[inline]
unsafe fn dot_sse<const N: usize>(win: &[f32], ht: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let half = N / 2;
    let wp = win.as_ptr();
    let tp = ht.as_ptr();
    let mut acc = _mm_setzero_ps();
    let full4 = half / 4;
    for i in 0..full4 {
        let k = i * 4;
        let left = _mm_loadu_ps(wp.add(k));
        // Load win[N-4-k..N-k] then reverse all 4 elements.
        let raw   = _mm_loadu_ps(wp.add(N - 1 - k - 3));
        // _MM_SHUFFLE(0,1,2,3) = 0x1B reverses 4 floats.
        let right = _mm_shuffle_ps(raw, raw, 0x1B);
        let pair  = _mm_add_ps(left, right);
        let taps  = _mm_loadu_ps(tp.add(k));
        // No FMA intrinsic required: the compiler fuses mul+add with -C opt-level=3.
        acc = _mm_add_ps(acc, _mm_mul_ps(taps, pair));
    }
    // Horizontal sum using hadd (SSE3).
    let s1 = _mm_hadd_ps(acc, acc);
    let s2 = _mm_hadd_ps(s1, s1);
    let mut total = _mm_cvtss_f32(s2);
    for k in full4 * 4..half {
        total += *tp.add(k) * (*wp.add(k) + *wp.add(N - 1 - k));
    }
    total += *tp.add(half) * *wp.add(half);
    total
}

// ============================================================================
// Batch decimation (public)
// ============================================================================

/// Bounds-checked symmetric dot for startup/tail samples (window extends
/// outside [0, len)).  Uses the same arithmetic order as the hot path so
/// that batch and streaming produce identical bytes.
#[inline(never)]
fn dot_sym_checked<const N: usize>(
    x: &[f32], center: usize, ht: &[f32],
) -> f32 {
    let half = N / 2;
    let len = x.len();
    let mut acc = 0.0f32;
    for k in 0..half {
        let li = center as isize - k as isize;
        let ri = center as isize - (N as isize - 1 - k as isize);
        let lv = if li >= 0 && (li as usize) < len { unsafe { *x.get_unchecked(li as usize) } } else { 0.0 };
        let rv = if ri >= 0 && (ri as usize) < len { unsafe { *x.get_unchecked(ri as usize) } } else { 0.0 };
        acc += ht[k] * (lv + rv);
    }
    let ci = center as isize - half as isize;
    let cv = if ci >= 0 && (ci as usize) < len { unsafe { *x.get_unchecked(ci as usize) } } else { 0.0 };
    acc + ht[half] * cv
}

/// (3+4) Inner decimation loop, monomorphised over N_TAPS, FACTOR, and output format.
/// MU=true → mu-law output (1 byte/sample); MU=false → s16le output (2 bytes/sample).
/// Assumes x is already pre-converted to f32.
fn decimate_sym_inner<const N: usize, const F: usize, const MU: bool>(
    x: &[f32], ht: &[f32], lut: &[u8; 65536], out: &mut Vec<u8>,
) {
    let total_out = x.len() / F;
    out.clear();
    out.reserve(if MU { total_out } else { total_out * 2 });

    // Startup: first outputs whose convolution window extends before x[0].
    // Need center >= N-1, i.e. n_out >= ceil((N-1)/F).
    let startup = ((N - 1) + F - 1) / F;
    let startup = startup.min(total_out);

    for n in 0..startup {
        let acc = dot_sym_checked::<N>(x, n * F, ht);
        if MU {
            out.push(lut[clamp_i16(acc) as u16 as usize]);
        } else {
            out.extend_from_slice(&clamp_i16(acc).to_le_bytes());
        }
    }

    // Main fast path: full window guaranteed inside x.
    for n in startup..total_out {
        let center = n * F;
        let ws = center + 1 - N;
        // SAFETY: ws = center-N+1 >= 0 (startup guarantees center >= N-1)
        //         ws+N = center+1 <= x.len() (total_out = x.len()/F, center = n*F < x.len()).
        let win = unsafe { std::slice::from_raw_parts(x.as_ptr().add(ws), N) };
        let acc = dot_sym::<N>(win, ht);
        if MU {
            // SAFETY: clamp_i16 returns i16, cast to u16 is always 0..65535.
            out.push(unsafe { *lut.get_unchecked(clamp_i16(acc) as u16 as usize) });
        } else {
            out.extend_from_slice(&clamp_i16(acc).to_le_bytes());
        }
    }
}

/// Generic fallback for tap counts / factors not in the specialised match.
fn decimate_generic(
    x: &[f32], taps: &[f32], factor: usize, lut: &[u8; 65536], out: &mut Vec<u8>,
) {
    let n = taps.len();
    let total = x.len() / factor;
    out.clear();
    out.reserve(total);
    for n_out in 0..total {
        let center = n_out * factor;
        let mut acc = 0.0f32;
        for k in 0..n {
            let idx = center as isize - k as isize;
            if idx >= 0 && (idx as usize) < x.len() {
                acc += taps[k] * x[idx as usize];
            }
        }
        out.push(lut[clamp_i16(acc) as u16 as usize]);
    }
}

fn decimate_to_output(input: &[i16], taps: &[f32], factor: usize, out_fmt: OutFmt, out: &mut Vec<u8>) {
    let lut = mulaw_lut();
    // (3) Pre-convert entire input to f32 once.
    let x: Vec<f32> = input.iter().map(|&s| s as f32).collect();
    let ht = fold_taps(taps);
    // (4) Dispatch to monomorphised specialisation.
    match (taps.len(), factor, out_fmt) {
        (63,  3, OutFmt::MuLaw) => decimate_sym_inner::<63,  3, true>(&x, &ht, lut, out),
        (63,  3, OutFmt::S16le) => decimate_sym_inner::<63,  3, false>(&x, &ht, lut, out),
        (127, 6, OutFmt::MuLaw) => decimate_sym_inner::<127, 6, true>(&x, &ht, lut, out),
        (127, 6, OutFmt::S16le) => decimate_sym_inner::<127, 6, false>(&x, &ht, lut, out),
        _                       => decimate_generic(&x, taps, factor, lut, out),
    }
}

/// Public batch API — mu-law output, unchanged signature.
pub fn decimate_to_mulaw(input: &[i16], taps: &[f32], factor: usize, out: &mut Vec<u8>) {
    decimate_to_output(input, taps, factor, OutFmt::MuLaw, out);
}

// ============================================================================
// Format / config (Python-facing path selection)
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct PathSpec {
    factor: usize,
    n_taps: usize,
    cutoff_hz: f32,
    in_sr: u32,
    out_fmt: OutFmt,
}

fn select_path(
    in_format: &str, in_sr: u32, out_format: &str, out_sr: u32, channels: u32,
) -> Option<PathSpec> {
    if channels != 1 { return None; }
    if in_format != "s16le" { return None; }
    let out_fmt = OutFmt::from_str(out_format)?;
    match (in_sr, out_sr) {
        (48_000, 16_000) => Some(PathSpec { factor: 3,  n_taps: 63,  cutoff_hz: 7_600.0, in_sr, out_fmt }),
        (48_000,  8_000) => Some(PathSpec { factor: 6,  n_taps: 127, cutoff_hz: 3_800.0, in_sr, out_fmt }),
        _ => None,
    }
}

fn build_taps(spec: &PathSpec) -> Vec<f32> {
    design_lowpass_fir(spec.n_taps, spec.cutoff_hz / spec.in_sr as f32)
}

#[cfg(feature = "python")]
fn bytes_to_i16(buf: &[u8]) -> PyResult<&[i16]> {
    if buf.len() % 2 != 0 {
        return Err(PyValueError::new_err("input byte length must be even (s16le)"));
    }
    let ptr = buf.as_ptr() as *const i16;
    Ok(unsafe { std::slice::from_raw_parts(ptr, buf.len() / 2) })
}

// ============================================================================
// Streaming resampler (stores f32 history, uses optimised dot)
// ============================================================================

struct Streaming {
    half_taps: Vec<f32>,
    n_taps: usize,
    factor: usize,
    out_fmt: OutFmt,
    total_in_abs: i64,  // absolute count of input samples seen so far
    buf_start_abs: i64, // absolute index of buf[0]
    buf: Vec<f32>,      // f32 history; kept trimmed to last (n_taps-1) samples
    next_out_abs: i64,  // absolute index (input-sample space) of next output's centre
}

impl Streaming {
    fn new(spec: &PathSpec) -> Self {
        let taps = build_taps(spec);
        let half_taps = fold_taps(&taps);
        let n_taps = taps.len();
        Self {
            half_taps,
            n_taps,
            factor: spec.factor,
            out_fmt: spec.out_fmt,
            total_in_abs: 0,
            buf_start_abs: 0,
            buf: Vec::new(),
            next_out_abs: 0,
        }
    }

    fn feed(&mut self, samples: &[i16], out: &mut Vec<u8>) {
        let lut = mulaw_lut();
        // (3) Append pre-converted f32 samples.
        self.buf.extend(samples.iter().map(|&s| s as f32));
        self.total_in_abs += samples.len() as i64;

        loop {
            let c = self.next_out_abs;
            if c >= self.total_in_abs { break; }

            let acc = if c >= self.n_taps as i64 - 1 {
                // Fast path: full window is in buf and all indices ≥ 0.
                let win_abs_start = c - self.n_taps as i64 + 1;
                let win_buf_start = (win_abs_start - self.buf_start_abs) as usize;
                // SAFETY: invariants (see comments in feed_window_safe below) guarantee
                // win_buf_start + n_taps <= buf.len().
                let win = unsafe {
                    std::slice::from_raw_parts(self.buf.as_ptr().add(win_buf_start), self.n_taps)
                };
                match self.n_taps {
                    63  => dot_sym::<63>(win, &self.half_taps),
                    127 => dot_sym::<127>(win, &self.half_taps),
                    _   => self.dot_checked(c),
                }
            } else {
                // Slow path: startup (window extends before input start).
                self.dot_checked(c)
            };

            push_sample(acc, out, lut, self.out_fmt);
            self.next_out_abs += self.factor as i64;
        }

        // Trim buf: keep only the tail needed for the next window.
        let keep_from = (self.next_out_abs - self.n_taps as i64 + 1).max(0);
        let drop = (keep_from - self.buf_start_abs).max(0) as usize;
        let drop = drop.min(self.buf.len());
        if drop > 0 {
            self.buf.drain(..drop);
            self.buf_start_abs += drop as i64;
        }
    }

    fn flush(&mut self, out: &mut Vec<u8>) {
        let lut = mulaw_lut();
        let total_out = self.total_in_abs / self.factor as i64;
        while self.next_out_abs < total_out * self.factor as i64 {
            let c = self.next_out_abs;
            let acc = self.dot_checked(c);
            push_sample(acc, out, lut, self.out_fmt);
            self.next_out_abs += self.factor as i64;
        }
        self.buf.clear();
    }

    /// Bounds-checked symmetric dot product — used for startup, tail, and flush.
    /// Same arithmetic order as dot_sym_checked so results agree with the fast path.
    #[inline(never)]
    fn dot_checked(&self, c: i64) -> f32 {
        let half = self.n_taps / 2;
        let mut acc = 0.0f32;
        for k in 0..half {
            let lv = self.get_sample(c - k as i64);
            let rv = self.get_sample(c - (self.n_taps as i64 - 1 - k as i64));
            acc += self.half_taps[k] * (lv + rv);
        }
        acc + self.half_taps[half] * self.get_sample(c - half as i64)
    }

    #[inline(always)]
    fn get_sample(&self, abs: i64) -> f32 {
        if abs < 0 || abs >= self.total_in_abs { return 0.0; }
        let idx = (abs - self.buf_start_abs) as usize;
        unsafe { *self.buf.get_unchecked(idx) }
    }
}

// ============================================================================
// PyO3 bindings (feature-gated)
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
fn is_supported(in_format: &str, in_sr: u32, out_format: &str, out_sr: u32, channels: u32) -> bool {
    select_path(in_format, in_sr, out_format, out_sr, channels).is_some()
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (input, in_format="s16le", in_sr=48_000, out_format="mulaw", out_sr=8_000, channels=1))]
fn convert_bytes(
    py: Python<'_>, input: &[u8],
    in_format: &str, in_sr: u32, out_format: &str, out_sr: u32, channels: u32,
) -> PyResult<PyObject> {
    let spec = select_path(in_format, in_sr, out_format, out_sr, channels).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unsupported combo: {}@{} ch={} -> {}@{}", in_format, in_sr, channels, out_format, out_sr
        ))
    })?;
    let pcm = bytes_to_i16(input)?;
    let taps = build_taps(&spec);
    let mut out = Vec::with_capacity(pcm.len() / spec.factor);
    py.allow_threads(|| { decimate_to_output(pcm, &taps, spec.factor, spec.out_fmt, &mut out); });
    Ok(PyBytes::new_bound(py, &out).into())
}

#[cfg(feature = "python")]
#[pyclass]
struct Resampler {
    inner: Streaming,
    finalized: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl Resampler {
    #[new]
    #[pyo3(signature = (in_format="s16le", in_sr=48_000, out_format="mulaw", out_sr=8_000, channels=1))]
    fn new(in_format: &str, in_sr: u32, out_format: &str, out_sr: u32, channels: u32) -> PyResult<Self> {
        let spec = select_path(in_format, in_sr, out_format, out_sr, channels).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unsupported combo: {}@{} ch={} -> {}@{}", in_format, in_sr, channels, out_format, out_sr
            ))
        })?;
        Ok(Self { inner: Streaming::new(&spec), finalized: false })
    }

    fn feed(&mut self, py: Python<'_>, input: &[u8]) -> PyResult<PyObject> {
        if self.finalized {
            return Err(PyTypeError::new_err("Resampler is finalized; create a new one"));
        }
        let pcm = bytes_to_i16(input)?;
        let bps = if self.inner.out_fmt == OutFmt::MuLaw { 1usize } else { 2 };
        let mut out = Vec::with_capacity((pcm.len() / self.inner.factor + 1) * bps);
        py.allow_threads(|| { self.inner.feed(pcm, &mut out); });
        Ok(PyBytes::new_bound(py, &out).into())
    }

    fn flush(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let mut out = Vec::new();
        py.allow_threads(|| { self.inner.flush(&mut out); });
        self.finalized = true;
        Ok(PyBytes::new_bound(py, &out).into())
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn flac_decode(py: Python<'_>, input: &[u8]) -> PyResult<(PyObject, u32, u32, u32)> {
    let decoded = py
        .allow_threads(|| flac::decode_flac_to_s16le(input))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bytes = PyBytes::new_bound(py, &decoded.pcm_s16le).into();
    Ok((bytes, decoded.sample_rate, decoded.channels, decoded.bits_per_sample))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (input, sample_rate, channels=1, compression_level=5))]
fn flac_encode(
    py: Python<'_>,
    input: &[u8],
    sample_rate: u32,
    channels: u32,
    compression_level: u8,
) -> PyResult<PyObject> {
    let bytes = py
        .allow_threads(|| {
            flac::encode_s16le_to_flac(input, sample_rate, channels, compression_level)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new_bound(py, &bytes).into())
}

#[cfg(feature = "python")]
#[pymodule]
fn _rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_supported, m)?)?;
    m.add_function(wrap_pyfunction!(convert_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(flac_decode, m)?)?;
    m.add_function(wrap_pyfunction!(flac_encode, m)?)?;
    m.add_class::<Resampler>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(n: usize) -> Vec<i16> {
        (0..n).map(|i| {
            let t = i as f32 / 48_000.0;
            clamp_i16(0.5 * (2.0 * std::f32::consts::PI * 400.0 * t).sin() * 32767.0)
        }).collect()
    }

    fn batch_mulaw(input: &[i16], spec: PathSpec) -> Vec<u8> {
        let taps = build_taps(&spec);
        let mut out = Vec::new();
        decimate_to_mulaw(input, &taps, spec.factor, &mut out);
        out
    }

    fn batch_s16le(input: &[i16], spec: PathSpec) -> Vec<u8> {
        let taps = build_taps(&spec);
        let mut out = Vec::new();
        decimate_to_output(input, &taps, spec.factor, OutFmt::S16le, &mut out);
        out
    }

    fn streaming_chunked(input: &[i16], spec: PathSpec, chunk: usize) -> Vec<u8> {
        let mut s = Streaming::new(&spec);
        let mut out = Vec::new();
        for c in input.chunks(chunk) { s.feed(c, &mut out); }
        s.flush(&mut out);
        out
    }

    #[test]
    fn streaming_matches_batch_48_to_8() {
        let spec = PathSpec { factor: 6, n_taps: 127, cutoff_hz: 3800.0, in_sr: 48_000, out_fmt: OutFmt::MuLaw };
        let sig = make_signal(48_000);
        let b = batch_mulaw(&sig, spec);
        for &chunk in &[1usize, 7, 128, 480, 1024, 4800, 48_000] {
            let s = streaming_chunked(&sig, spec, chunk);
            assert_eq!(b.len(), s.len(), "len mismatch chunk={chunk}");
            assert_eq!(b, s, "byte mismatch chunk={chunk}");
        }
    }

    #[test]
    fn streaming_matches_batch_48_to_16() {
        let spec = PathSpec { factor: 3, n_taps: 63, cutoff_hz: 7600.0, in_sr: 48_000, out_fmt: OutFmt::MuLaw };
        let sig = make_signal(48_000);
        let b = batch_mulaw(&sig, spec);
        for &chunk in &[1usize, 7, 128, 480, 1024, 4800, 48_000] {
            let s = streaming_chunked(&sig, spec, chunk);
            assert_eq!(b.len(), s.len(), "len mismatch chunk={chunk}");
            assert_eq!(b, s, "byte mismatch chunk={chunk}");
        }
    }

    #[test]
    fn streaming_matches_batch_s16le_48_to_8() {
        let spec = PathSpec { factor: 6, n_taps: 127, cutoff_hz: 3800.0, in_sr: 48_000, out_fmt: OutFmt::S16le };
        let sig = make_signal(48_000);
        let b = batch_s16le(&sig, spec);
        assert_eq!(b.len(), sig.len() / 6 * 2, "output must be 2 bytes/sample");
        for &chunk in &[1usize, 7, 128, 480, 1024, 4800, 48_000] {
            let s = streaming_chunked(&sig, spec, chunk);
            assert_eq!(b.len(), s.len(), "len mismatch chunk={chunk}");
            assert_eq!(b, s, "byte mismatch chunk={chunk}");
        }
    }

    #[test]
    fn streaming_matches_batch_s16le_48_to_16() {
        let spec = PathSpec { factor: 3, n_taps: 63, cutoff_hz: 7600.0, in_sr: 48_000, out_fmt: OutFmt::S16le };
        let sig = make_signal(48_000);
        let b = batch_s16le(&sig, spec);
        assert_eq!(b.len(), sig.len() / 3 * 2, "output must be 2 bytes/sample");
        for &chunk in &[1usize, 7, 128, 480, 1024, 4800, 48_000] {
            let s = streaming_chunked(&sig, spec, chunk);
            assert_eq!(b.len(), s.len(), "len mismatch chunk={chunk}");
            assert_eq!(b, s, "byte mismatch chunk={chunk}");
        }
    }
}
