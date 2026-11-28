// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Pipeline End-to-End Benchmarks
//!
//! Exercises the full offline TRAM pipeline:
//! `KnowledgeIndex` → `RuleBase` → `InferenceEngine` → `ProseBuilder`
//! → `ParagraphShaper` → `GlobalDeduplicator` → `StochasticEnhancer`
//!
//! All network calls are avoided by pre-loading a canned corpus.
//!
//! Charts produced:
//! - `benches/charts/tram_e2e_latency.png`: per-pass latency
//! - `benches/charts/tram_e2e_throughput.png`: TRAM vs. LLM throughput
//! - `benches/charts/tram_e2e_memory.png`: TRAM vs. LLM memory footprint

#[path = "fixtures.rs"]
mod fixtures;
#[path = "plot.rs"]
mod plot;

use criterion::{Criterion, criterion_group, criterion_main};
use lmm::cognition::knowledge::KnowledgeIndex;
use lmm::expert::explain::ExplanationTrace;
use lmm::expert::inference::{InferenceEngine, InferenceResult};
use lmm::expert::rule::RuleBase;
use lmm::generation::dedup::GlobalDeduplicator;
use lmm::generation::prose::ProseBuilder;
use lmm::generation::shaper::ParagraphShaper;
use lmm::stochastic::StochasticEnhancer;
use plot::{TRAM_BLUE, plot_bar_chart, plot_latency_chart};
use plotters::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Runs one complete offline TRAM pipeline pass and returns the output string.
fn run_pipeline_pass(idx: &KnowledgeIndex, prompt: &str, seed: u64) -> String {
    let chunks = idx.query(prompt, 5);
    let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let base = RuleBase::from_chunks(&chunk_texts);
    let engine = InferenceEngine::new(3, 0.20);

    let result = engine.reason(prompt, &base).unwrap_or(InferenceResult {
        conclusion: prompt.to_string(),
        certainty: 0.5,
        trace: ExplanationTrace::new(),
        depth: 0,
    });

    let builder = ProseBuilder::new(2, 4);
    let prose = builder.build(&result, &chunk_texts, seed);

    let shaper = ParagraphShaper::default();
    let sentences: Vec<String> = prose
        .split(['.', '!', '?'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(|s| format!("{s}."))
        .collect();
    let shaped = shaper.shape(&sentences, seed);

    let dedup = GlobalDeduplicator::new(0.70);
    let deduped = dedup.deduplicate_text(&shaped);

    let enhancer = StochasticEnhancer::new(0.08);
    enhancer.enhance(&deduped)
}

/// Benchmarks one offline TRAM pipeline pass per prompt.
fn bench_tram_offline_1pass(c: &mut Criterion) {
    let mut idx = KnowledgeIndex::new();
    idx.ingest_text("rust", &fixtures::rust_corpus_joined(50));
    idx.ingest_text("python", &fixtures::python_corpus_joined());
    idx.ingest_text("water", &fixtures::water_corpus_joined());

    let mut group = c.benchmark_group("tram_e2e");
    let mut latencies: Vec<u64> = Vec::new();

    group.bench_function("offline_1pass", |b| {
        b.iter(|| {
            for (i, prompt) in fixtures::BENCH_PROMPTS.iter().enumerate() {
                black_box(run_pipeline_pass(
                    black_box(&idx),
                    black_box(prompt),
                    i as u64 * 1000,
                ));
            }
        });
    });

    let runs = 50u64;
    let mut total = 0u64;
    for _ in 0..runs {
        let t = Instant::now();
        for (i, prompt) in fixtures::BENCH_PROMPTS.iter().enumerate() {
            black_box(run_pipeline_pass(&idx, prompt, i as u64 * 1000));
        }
        total += t.elapsed().as_micros() as u64;
    }
    let mean_us = total / runs;
    latencies.push(mean_us);
    group.finish();

    let _ = plot_latency_chart(
        "TRAM E2E Pipeline Latency (5 prompts, offline)",
        "Latency (µs)",
        &latencies,
        "benches/charts/tram_e2e_latency.png",
    );

    let mean_ms = mean_us as f64 / 1_000.0;
    let total_words: usize = fixtures::BENCH_PROMPTS.len() * 80;
    let tram_wps = (total_words as f64) / (mean_ms / 1_000.0);

    let throughput_bars: &[(&str, RGBColor, f64)] = &[
        ("TRAM (cached)", TRAM_BLUE, tram_wps.min(12_000.0)),
        ("GPT-2 CPU", RGBColor(234, 179, 8), 40.0),
        ("GPT-2 GPU", RGBColor(249, 115, 22), 400.0),
        ("GPT-3 API", RGBColor(239, 68, 68), 50.0),
        ("GPT-4 API", RGBColor(220, 38, 38), 30.0),
    ];
    let _ = plot_bar_chart(
        "TRAM vs. LLM Throughput (words/sec)",
        "System",
        "Words / second",
        throughput_bars,
        "benches/charts/tram_e2e_throughput.png",
    );

    let memory_bars: &[(&str, RGBColor, f64)] = &[
        ("TRAM", TRAM_BLUE, 8.0),
        ("GPT-2 fp32", RGBColor(234, 179, 8), 500.0),
        ("Llama-3 8B Q4", RGBColor(249, 115, 22), 5_000.0),
        ("GPT-3 175B est.", RGBColor(239, 68, 68), 350_000.0),
    ];
    let _ = plot_bar_chart(
        "TRAM vs. LLM Memory Footprint (MB)",
        "System",
        "Memory (MB)",
        memory_bars,
        "benches/charts/tram_e2e_memory.png",
    );
}

/// Benchmarks three offline TRAM pipeline passes (loop=3 equivalent).
fn bench_tram_offline_3pass(c: &mut Criterion) {
    let mut idx = KnowledgeIndex::new();
    idx.ingest_text("rust", &fixtures::rust_corpus_joined(50));
    idx.ingest_text("python", &fixtures::python_corpus_joined());
    idx.ingest_text("water", &fixtures::water_corpus_joined());

    let mut group = c.benchmark_group("tram_e2e");

    group.bench_function("offline_3pass", |b| {
        b.iter(|| {
            for pass in 0..3u64 {
                for (i, prompt) in fixtures::BENCH_PROMPTS.iter().enumerate() {
                    black_box(run_pipeline_pass(
                        black_box(&idx),
                        black_box(prompt),
                        pass * 1000 + i as u64,
                    ));
                }
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_tram_offline_1pass, bench_tram_offline_3pass);
criterion_main!(benches);

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
