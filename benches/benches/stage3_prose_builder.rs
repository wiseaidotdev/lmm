// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Stage 3: Prose Builder & Coherence Scorer Benchmarks
//!
//! Measures `ProseBuilder::build` for 1-step and 5-step proof traces, and
//! `CoherenceScorer::reorder` for 4, 8, and 16-sentence inputs.
//!
//! Chart produced:
//! - `benches/charts/prose_builder.png`: per-iteration latency

#[path = "fixtures.rs"]
mod fixtures;
#[path = "plot.rs"]
mod plot;

use criterion::{Criterion, criterion_group, criterion_main};
use lmm::expert::explain::ExplanationTrace;
use lmm::expert::inference::InferenceResult;
use lmm::generation::coherence::CoherenceScorer;
use lmm::generation::prose::ProseBuilder;
use plot::plot_latency_chart;
use std::hint::black_box;
use std::time::Instant;

/// Builds an `InferenceResult` from a slice of `(conclusion, certainty)` pairs.
fn make_result(steps: &[(&str, f64)]) -> InferenceResult {
    let mut trace = ExplanationTrace::new();
    for (conclusion, certainty) in steps {
        trace.push(lmm::expert::explain::ProofStep::told(
            conclusion.to_string(),
        ));
        let _ = certainty;
    }
    let (top_conc, top_cf) = steps.last().copied().unwrap_or(("unknown", 0.5));
    InferenceResult {
        conclusion: top_conc.to_string(),
        certainty: top_cf,
        trace,
        depth: steps.len(),
    }
}

/// Benchmarks `ProseBuilder::build` with a 1-step trace.
fn bench_prose_build_1step(c: &mut Criterion) {
    let result = make_result(fixtures::PROOF_STEPS_1);
    let chunks: Vec<&str> = fixtures::RUST_CORPUS.iter().take(5).copied().collect();
    let builder = ProseBuilder::new(2, 4);
    let mut group = c.benchmark_group("prose_build");

    group.bench_function("1_step", |b| {
        b.iter(|| black_box(builder.build(black_box(&result), black_box(&chunks), 12345)));
    });
    group.finish();
}

/// Benchmarks `ProseBuilder::build` with a 5-step trace.
fn bench_prose_build_5steps(c: &mut Criterion) {
    let result = make_result(fixtures::PROOF_STEPS_5);
    let chunks: Vec<&str> = fixtures::RUST_CORPUS.iter().take(5).copied().collect();
    let builder = ProseBuilder::new(2, 4);
    let mut group = c.benchmark_group("prose_build");

    group.bench_function("5_steps", |b| {
        b.iter(|| black_box(builder.build(black_box(&result), black_box(&chunks), 99999)));
    });
    group.finish();
}

/// Benchmarks `CoherenceScorer::reorder` for 4, 8, and 12-sentence inputs.
fn bench_coherence_reorder(c: &mut Criterion) {
    let scorer = CoherenceScorer;
    let sizes: &[usize] = &[4, 8, 12];
    let mut group = c.benchmark_group("coherence_reorder");
    let mut latencies: Vec<u64> = Vec::new();

    for &n in sizes {
        let sentences = fixtures::rust_sentences(n);
        let label = format!("reorder_{n}");
        group.bench_function(&label, |b| {
            b.iter(|| black_box(scorer.reorder(black_box(&sentences))));
        });

        let runs = 500u64;
        let mut total = 0u64;
        for _ in 0..runs {
            let t = Instant::now();
            black_box(scorer.reorder(&sentences));
            total += t.elapsed().as_micros() as u64;
        }
        latencies.push(total / runs);
    }
    group.finish();

    let _ = plot_latency_chart(
        "ProseBuilder & CoherenceScorer Latency",
        "Latency (µs)",
        &latencies,
        "benches/charts/prose_builder.png",
    );
}

criterion_group!(
    benches,
    bench_prose_build_1step,
    bench_prose_build_5steps,
    bench_coherence_reorder
);
criterion_main!(benches);

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
