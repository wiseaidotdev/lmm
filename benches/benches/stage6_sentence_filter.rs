// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Stage 6: Sentence Quality Filter Benchmarks
//!
//! Measures `SentenceFilter::is_acceptable` on accept and reject sentence sets,
//! demonstrating the near-zero PHF-backed overhead.
//!
//! Chart produced:
//! - `benches/charts/sentence_filter.png`: per-iteration latency

#[path = "fixtures.rs"]
mod fixtures;
#[path = "plot.rs"]
mod plot;

use criterion::{Criterion, criterion_group, criterion_main};
use lmm::generation::filter::SentenceFilter;
use plot::plot_latency_chart;
use std::hint::black_box;
use std::time::Instant;

/// Benchmarks `SentenceFilter::is_acceptable` on the RUST_CORPUS (all accept).
fn bench_filter_accept(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentence_filter");
    let mut latencies: Vec<u64> = Vec::new();

    group.bench_function("accept_50_sentences", |b| {
        b.iter(|| {
            for s in fixtures::RUST_CORPUS.iter().take(50) {
                black_box(SentenceFilter::is_acceptable(black_box(s)));
            }
        });
    });

    let runs = 500u64;
    let mut total = 0u64;
    for _ in 0..runs {
        let t = Instant::now();
        for s in fixtures::RUST_CORPUS.iter().take(50) {
            black_box(SentenceFilter::is_acceptable(s));
        }
        total += t.elapsed().as_micros() as u64;
    }
    latencies.push(total / runs);

    group.finish();

    let _ = plot_latency_chart(
        "SentenceFilter Latency - Accept vs. Reject",
        "Latency (µs)",
        &latencies,
        "benches/charts/sentence_filter.png",
    );
}

/// Benchmarks `SentenceFilter::is_acceptable` on JUNK_SENTENCES (all reject).
fn bench_filter_reject(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentence_filter");

    group.bench_function("reject_junk_sentences", |b| {
        b.iter(|| {
            for s in fixtures::JUNK_SENTENCES {
                black_box(SentenceFilter::is_acceptable(black_box(s)));
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_filter_accept, bench_filter_reject);
criterion_main!(benches);

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
