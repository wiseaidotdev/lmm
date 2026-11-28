// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Stage 5: Stochastic Enhancer Benchmarks
//!
//! Measures `StochasticEnhancer::enhance` at three substitution probabilities
//! and plots throughput (words/sec) against probability.
//!
//! Chart produced:
//! - `benches/charts/stochastic_throughput.png`: probability vs. words/sec

#[path = "fixtures.rs"]
mod fixtures;
#[path = "plot.rs"]
mod plot;

use criterion::{Criterion, criterion_group, criterion_main};
use lmm::stochastic::StochasticEnhancer;
use plot::plot_throughput_chart;
use std::hint::black_box;
use std::time::Instant;

/// Benchmarks `StochasticEnhancer::enhance` at probabilities 0.08, 0.30, 0.50.
fn bench_stochastic_enhance(c: &mut Criterion) {
    let probabilities: &[f64] = &[0.08, 0.30, 0.50];
    let text = fixtures::rust_corpus_joined(10);
    let word_count = text.split_whitespace().count();

    let mut group = c.benchmark_group("stochastic_enhance");
    let mut prob_as_pct: Vec<usize> = Vec::new();
    let mut throughputs: Vec<f64> = Vec::new();

    for &p in probabilities {
        let enhancer = StochasticEnhancer::new(p);
        let label = format!("p{}", (p * 100.0) as u32);
        group.bench_function(&label, |b| {
            b.iter(|| black_box(enhancer.enhance(black_box(&text))));
        });

        let runs = 200u64;
        let mut total_ns = 0u64;
        for _ in 0..runs {
            let t = Instant::now();
            black_box(enhancer.enhance(&text));
            total_ns += t.elapsed().as_nanos() as u64;
        }
        let mean_secs = (total_ns / runs) as f64 / 1_000_000_000.0;
        let wps = word_count as f64 / mean_secs;
        prob_as_pct.push((p * 100.0) as usize);
        throughputs.push(wps);
    }
    group.finish();

    let _ = plot_throughput_chart(
        "StochasticEnhancer Throughput vs. Substitution Probability",
        "Probability (%)",
        "Words / second",
        &prob_as_pct,
        &throughputs,
        "benches/charts/stochastic_throughput.png",
    );
}

criterion_group!(benches, bench_stochastic_enhance);
criterion_main!(benches);

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
