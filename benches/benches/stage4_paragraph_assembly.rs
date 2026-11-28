// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Stage 4: Paragraph Assembly Benchmarks
//!
//! Measures `GlobalDeduplicator::deduplicate` and `ParagraphShaper::shape`
//! for increasing sentence counts.
//!
//! Chart produced:
//! - `benches/charts/paragraph_assembly.png`: sentence count vs. dedup latency

#[path = "fixtures.rs"]
mod fixtures;
#[path = "plot.rs"]
mod plot;

use criterion::{Criterion, criterion_group, criterion_main};
use lmm::generation::dedup::GlobalDeduplicator;
use lmm::generation::shaper::ParagraphShaper;
use plot::{TRAM_BLUE, plot_scaling_chart};
use std::hint::black_box;
use std::time::Instant;

/// Benchmarks `GlobalDeduplicator::deduplicate` for N sentences.
fn bench_global_dedup(c: &mut Criterion) {
    let sizes: &[usize] = &[5, 10, 20, 40];
    let dedup = GlobalDeduplicator::new(0.70);
    let mut group = c.benchmark_group("global_dedup");
    let mut scaling_pts: Vec<(usize, f64)> = Vec::new();

    for &n in sizes {
        let sentences = fixtures::rust_sentences(n);
        let label = format!("dedup_{n}");
        group.bench_function(&label, |b| {
            b.iter(|| black_box(dedup.deduplicate(black_box(sentences.clone()))));
        });

        let runs = 200u64;
        let mut total = 0u64;
        for _ in 0..runs {
            let t = Instant::now();
            black_box(dedup.deduplicate(sentences.clone()));
            total += t.elapsed().as_micros() as u64;
        }
        scaling_pts.push((n, (total / runs) as f64));
    }
    group.finish();

    let _ = plot_scaling_chart(
        "GlobalDeduplicator Latency vs. Sentence Count",
        "Sentences",
        "Latency (µs)",
        &[("TRAM dedup", TRAM_BLUE, scaling_pts)],
        "benches/charts/paragraph_assembly.png",
    );
}

/// Benchmarks `ParagraphShaper::shape` for N sentences.
fn bench_paragraph_shaper(c: &mut Criterion) {
    let sizes: &[usize] = &[5, 10, 20];
    let shaper = ParagraphShaper::default();
    let mut group = c.benchmark_group("paragraph_shaper");

    for &n in sizes {
        let sentences = fixtures::rust_sentences(n);
        let label = format!("shape_{n}");
        group.bench_function(&label, |b| {
            b.iter(|| black_box(shaper.shape(black_box(&sentences), black_box(42u64))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_global_dedup, bench_paragraph_shaper);
criterion_main!(benches);

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
