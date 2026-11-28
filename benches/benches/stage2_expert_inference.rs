// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Stage 2: Expert Inference Benchmarks
//!
//! Measures certainty-factor combination, rule-base construction, and
//! backward-chaining inference at varying depths.
//!
//! Charts produced:
//! - `benches/charts/expert_inference.png`: per-iteration latency
//! - `benches/charts/expert_depth.png`: inference latency vs. depth

#[path = "fixtures.rs"]
mod fixtures;
#[path = "plot.rs"]
mod plot;

use criterion::{Criterion, criterion_group, criterion_main};
use lmm::expert::inference::InferenceEngine;
use lmm::expert::rule::RuleBase;
use lmm::expert::uncertainty::{CertaintyFactor, combine_all};
use plot::{TRAM_BLUE, plot_latency_chart, plot_scaling_chart};
use plotters::prelude::GREEN;
use std::hint::black_box;
use std::time::Instant;

/// Benchmarks `combine_all` over increasing numbers of certainty factors.
fn bench_cf_combination(c: &mut Criterion) {
    let sizes: &[usize] = &[2, 5, 10, 20];
    let mut group = c.benchmark_group("cf_combination");

    for &n in sizes {
        let cfs: Vec<CertaintyFactor> = (0..n)
            .map(|i| CertaintyFactor::new(0.1 + (i as f64) * 0.03))
            .collect();
        let label = format!("combine_{n}");
        group.bench_function(&label, |b| {
            b.iter(|| black_box(combine_all(black_box(&cfs))));
        });
    }
    group.finish();
}

/// Benchmarks `RuleBase::from_chunks` rule extraction.
fn bench_rule_base_build(c: &mut Criterion) {
    let chunks: Vec<&str> = fixtures::RUST_CORPUS.iter().take(20).copied().collect();
    let mut group = c.benchmark_group("rule_base_build");

    group.bench_function("from_20_chunks", |b| {
        b.iter(|| black_box(RuleBase::from_chunks(black_box(&chunks))));
    });
    group.finish();
}

/// Benchmarks `InferenceEngine::reason` at max_depth 1-5.
fn bench_backward_chain(c: &mut Criterion) {
    let chunks: Vec<&str> = fixtures::RUST_CORPUS.iter().take(20).copied().collect();
    let base = RuleBase::from_chunks(&chunks);
    let depths: &[usize] = &[1, 2, 3, 4, 5];
    let mut group = c.benchmark_group("backward_chain");
    let mut latencies: Vec<u64> = Vec::new();
    let mut depth_pts: Vec<(usize, f64)> = Vec::new();

    for &d in depths {
        let engine = InferenceEngine::new(d, 0.20);
        let label = format!("depth_{d}");
        group.bench_function(&label, |b| {
            b.iter(|| black_box(engine.reason(black_box("rust memory safety"), black_box(&base))));
        });

        let runs = 200u64;
        let mut total = 0u64;
        for _ in 0..runs {
            let t = Instant::now();
            black_box(engine.reason("rust memory safety", &base));
            total += t.elapsed().as_micros() as u64;
        }
        let mean_us = total / runs;
        latencies.push(mean_us);
        depth_pts.push((d, mean_us as f64));
    }
    group.finish();

    let _ = plot_latency_chart(
        "Expert Inference Latency (depth 1-5)",
        "Latency (µs)",
        &latencies,
        "benches/charts/expert_inference.png",
    );

    let _ = plot_scaling_chart(
        "Expert Inference Latency vs. Max Depth",
        "Max backward-chain depth",
        "Latency (µs)",
        &[
            ("TRAM inference", TRAM_BLUE, depth_pts.clone()),
            (
                "Linear reference",
                GREEN,
                depth_pts.iter().map(|&(x, _)| (x, x as f64)).collect(),
            ),
        ],
        "benches/charts/expert_depth.png",
    );
}

criterion_group!(
    benches,
    bench_cf_combination,
    bench_rule_base_build,
    bench_backward_chain
);
criterion_main!(benches);

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
