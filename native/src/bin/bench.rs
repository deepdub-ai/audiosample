use audiosample_rs::*;
use std::time::Instant;

fn main() {
    let n = 30 * 48_000;
    let mut input = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    for i in 0..n {
        phase += 2.0 * std::f32::consts::PI * (200.0 + (i % 1000) as f32 * 5.0) / 48_000.0;
        input.push((0.5 * 32767.0 * phase.sin()) as i16);
    }
    let in_secs = input.len() as f64 / 48_000.0;

    let cases: &[(&str, usize, f32, usize)] = &[
        ("48k -> 16k", 63, 7600.0 / 48_000.0, 3),
        ("48k ->  8k", 127, 3800.0 / 48_000.0, 6),
    ];

    for &(name, n_taps, cutoff, factor) in cases {
        let taps = design_lowpass_fir(n_taps, cutoff);
        bench(name, &input, &taps, factor, in_secs, 200);
    }
}

fn bench(name: &str, input: &[i16], taps: &[f32], factor: usize, in_secs: f64, iters: usize) {
    let mut out = Vec::with_capacity(input.len() / factor);
    decimate_to_mulaw(input, taps, factor, &mut out);
    let t0 = Instant::now();
    for _ in 0..iters {
        decimate_to_mulaw(input, taps, factor, &mut out);
    }
    let secs = t0.elapsed().as_secs_f64();
    let ns_per_in = secs * 1e9 / (input.len() as f64 * iters as f64);
    let realtime = in_secs * iters as f64 / secs;
    let outputs = out.len() as f64 * iters as f64;
    let cy_per_out = secs * 3e9 / outputs; // assume ~3GHz
    println!(
        "{:<14} | taps={:3} factor={} | {:6.2} ns/in | {:6.1} ns/out (~{:5.0} cy/out @3GHz) | {:7.0}x rt",
        name,
        taps.len(),
        factor,
        ns_per_in,
        secs * 1e9 / outputs,
        cy_per_out,
        realtime,
    );
}
