// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to these terms.

//! # Intent-Aware Content Filter
//!
//! Scores and filters knowledge base chunks based on question intent.

use crate::cognition::thought::classifier::ReasoningMode;
use phf::phf_set;

static ADVANTAGE_KEYWORDS: phf::Set<&'static str> = phf_set! {
    "advantage", "benefit", "pro", "strength", "merit", "upside", "positive",
    "feature", "capability", "power", "excellent", "superior", "effective",
    "efficient", "fast", "speed", "performance", "reliable", "robust",
    "memory safety", "type safety", "concurrency", "ownership", "borrow checker",
    "zero-cost", "guarantee", "prevent", "without", "safe", "secure",
};

static CAUSAL_KEYWORDS: phf::Set<&'static str> = phf_set! {
    "because", "since", "due to", "as a result", "therefore", "thus", "hence",
    "consequently", "cause", "reason", "why", "leads to", "results in",
    "design", "ownership", "borrow", "memory", "prevent", "ensure",
};

static COMPARISON_KEYWORDS: phf::Set<&'static str> = phf_set! {
    "versus", "vs", "compared", "difference", "similar", "unlike", "whereas",
    "while", "contrast", "between", "against", "alternative", "option",
};

static DEFINITIONAL_KEYWORDS: phf::Set<&'static str> = phf_set! {
    "definition", "defined", "refers", "means", "concept", "term", "is a",
    "is an", "represents", "denotes", "signifies", "example", "instance",
};

static PROCEDURAL_KEYWORDS: phf::Set<&'static str> = phf_set! {
    "step", "how", "to", "install", "setup", "configure", "use", "implement",
    "create", "build", "make", "write", "code", "generate", "method",
    "procedure", "process", "approach", "technique",
};

pub struct IntentFilter;

impl IntentFilter {
    pub fn score_chunk(chunk: &str, mode: ReasoningMode) -> f64 {
        let lower = chunk.to_ascii_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        let keyword_set = match mode {
            ReasoningMode::Evaluative => &ADVANTAGE_KEYWORDS,
            ReasoningMode::Causal => &CAUSAL_KEYWORDS,
            ReasoningMode::Analogical => &COMPARISON_KEYWORDS,
            ReasoningMode::Definitional => &DEFINITIONAL_KEYWORDS,
            ReasoningMode::Procedural => &PROCEDURAL_KEYWORDS,
            _ => return 0.5,
        };

        let keyword_matches = words.iter().filter(|w| keyword_set.contains(*w)).count();
        let match_ratio = keyword_matches as f64 / words.len() as f64;

        let bonus = if match_ratio > 0.0 {
            match_ratio * 2.0
        } else {
            0.0
        };

        (0.3 + bonus).clamp(0.0, 1.0)
    }

    pub fn filter_and_rank(chunks: &[String], mode: ReasoningMode) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = chunks
            .iter()
            .map(|c| (c.clone(), Self::score_chunk(c, mode)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
    }

    pub fn select_top_chunks(chunks: &[String], mode: ReasoningMode, top_k: usize) -> Vec<String> {
        let scored = Self::filter_and_rank(chunks, mode);
        scored.into_iter().take(top_k).map(|(c, _)| c).collect()
    }

    pub fn select_top_chunks_with_deep_mode(
        chunks: &[String],
        mode: ReasoningMode,
        top_k: usize,
        deep_mode: bool,
    ) -> Vec<String> {
        if deep_mode {
            let scored = Self::filter_and_rank(chunks, mode);
            scored
                .into_iter()
                .filter(|(_, score)| *score >= 0.2)
                .take(top_k)
                .map(|(c, _)| c)
                .collect()
        } else {
            Self::select_top_chunks(chunks, mode, top_k)
        }
    }
}
