// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Stage 1: Knowledge Index Benchmarks
//!
//! Measures the throughput and latency of [`KnowledgeIndex`] ingestion and
//! top-k retrieval across increasing corpus sizes and k values.
//!
//! Charts produced:
//! - `benches/charts/kb_ingest.png`: O(n) ingest scaling
//! - `benches/charts/kb_retrieval.png`: retrieval latency vs. k

#[path = "fixtures.rs"]
mod fixtures;
#[path = "plot.rs"]
mod plot;

use criterion::{Criterion, criterion_group, criterion_main};
use lmm::cognition::knowledge::KnowledgeIndex;
use plot::{TRAM_BLUE, plot_scaling_chart};
use std::hint::black_box;
use std::time::Instant;

/// Benchmarks `KnowledgeIndex::ingest_text` for corpus sizes [10, 50, 100, 250, 500].
fn bench_kb_ingest(c: &mut Criterion) {
    let sizes: &[usize] = &[10, 50, 100, 250, 500];
    let mut group = c.benchmark_group("kb_ingest");
    let mut latencies: Vec<u64> = Vec::new();
    let mut scaling_pts: Vec<(usize, f64)> = Vec::new();

    for &n in sizes {
        let text = fixtures::rust_corpus_joined(n);
        let label = format!("ingest_n{n}");
        group.bench_function(&label, |b| {
            b.iter(|| {
                let mut idx = KnowledgeIndex::new();
                idx.ingest_text("bench", black_box(&text));
                black_box(idx)
            });
        });

        let mut total = 0u64;
        let runs = 50u64;
        for _ in 0..runs {
            let t = Instant::now();
            let mut idx = KnowledgeIndex::new();
            idx.ingest_text("bench", &text);
            black_box(idx);
            total += t.elapsed().as_micros() as u64;
        }
        let mean_us = total / runs;
        latencies.push(mean_us);
        scaling_pts.push((n, mean_us as f64));
    }
    group.finish();

    let _ = plot_scaling_chart(
        "KnowledgeIndex Ingest Latency vs. Corpus Size",
        "Sentences ingested",
        "Latency (µs)",
        &[("TRAM KB ingest", TRAM_BLUE, scaling_pts)],
        "benches/charts/kb_ingest.png",
    );
}

/// Benchmarks `KnowledgeIndex::query` for k values [1, 3, 5, 10].
fn bench_kb_retrieval(c: &mut Criterion) {
    let ks: &[usize] = &[1, 3, 5, 10];
    let text = fixtures::rust_corpus_joined(50);
    let mut idx = KnowledgeIndex::new();
    idx.ingest_text("bench", &text);

    let mut group = c.benchmark_group("kb_retrieval");
    let mut scaling_pts: Vec<(usize, f64)> = Vec::new();

    for &k in ks {
        let label = format!("query_k{k}");
        group.bench_function(&label, |b| {
            b.iter(|| black_box(idx.query(black_box("rust memory safety"), k)));
        });

        let runs = 200u64;
        let mut total = 0u64;
        for _ in 0..runs {
            let t = Instant::now();
            black_box(idx.query("rust memory safety", k));
            total += t.elapsed().as_micros() as u64;
        }
        scaling_pts.push((k, (total / runs) as f64));
    }
    group.finish();

    let _ = plot_scaling_chart(
        "KnowledgeIndex Retrieval Latency vs. k",
        "Top-k retrieved",
        "Latency (µs)",
        &[("TRAM KB retrieval", TRAM_BLUE, scaling_pts)],
        "benches/charts/kb_retrieval.png",
    );
}

criterion_group!(benches, bench_kb_ingest, bench_kb_retrieval);
criterion_main!(benches);

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
