// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # Sentence Relevance Scorer
//!
//! Scores individual sentences for relevance to question intent using TF-IDF style scoring.

use crate::cognition::thought::classifier::ReasoningMode;
use phf::phf_set;
use std::collections::HashMap;

static STOP_WORDS: phf::Set<&'static str> = phf_set! {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "its", "it", "this", "that",
    "these", "those", "not", "no", "nor", "so", "yet", "both", "also", "as", "if", "than",
    "then", "from", "into", "through", "during", "before", "after", "above", "below",
    "between", "each", "every", "any", "all",
};

pub struct RelevanceScorer;

impl RelevanceScorer {
    pub fn extract_terms(text: &str) -> Vec<String> {
        text.to_ascii_lowercase()
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(w.as_str()))
            .collect()
    }

    pub fn term_frequency(terms: &[String]) -> HashMap<String, f64> {
        let mut tf: HashMap<String, f64> = HashMap::new();
        let total = terms.len() as f64;

        for term in terms {
            *tf.entry(term.clone()).or_insert(0.0) += 1.0;
        }

        for freq in tf.values_mut() {
            *freq /= total;
        }

        tf
    }

    pub fn score_sentence(sentence: &str, query_terms: &[String], mode: ReasoningMode) -> f64 {
        let sentence_terms = Self::extract_terms(sentence);
        if sentence_terms.is_empty() {
            return 0.0;
        }

        let sentence_tf = Self::term_frequency(&sentence_terms);

        let mut score = 0.0;
        for query_term in query_terms {
            if let Some(&tf) = sentence_tf.get(query_term) {
                score += tf;
            }
        }

        let intent_bonus = Self::intent_bonus(sentence, mode);
        let length_penalty = if sentence.split_whitespace().count() < 8 {
            0.5
        } else {
            1.0
        };

        (score + intent_bonus) * length_penalty
    }

    fn intent_bonus(sentence: &str, mode: ReasoningMode) -> f64 {
        let lower = sentence.to_ascii_lowercase();
        match mode {
            ReasoningMode::Evaluative => {
                let has_pro = lower.contains("advantage")
                    || lower.contains("benefit")
                    || lower.contains("strength")
                    || lower.contains("positive");
                let has_con = lower.contains("disadvantage")
                    || lower.contains("drawback")
                    || lower.contains("weakness")
                    || lower.contains("negative");
                if has_pro || has_con { 0.3 } else { 0.0 }
            }
            ReasoningMode::Causal => {
                let has_causal = lower.contains("because")
                    || lower.contains("since")
                    || lower.contains("due to")
                    || lower.contains("reason")
                    || lower.contains("cause");
                if has_causal { 0.3 } else { 0.0 }
            }
            ReasoningMode::Analogical => {
                let has_comparison = lower.contains("versus")
                    || lower.contains("compared")
                    || lower.contains("difference")
                    || lower.contains("similar");
                if has_comparison { 0.3 } else { 0.0 }
            }
            ReasoningMode::Definitional => {
                let has_def = lower.contains("definition")
                    || lower.contains("defined")
                    || lower.contains("refers")
                    || lower.contains("means");
                if has_def { 0.3 } else { 0.0 }
            }
            ReasoningMode::Procedural => {
                let has_proc = lower.contains("step")
                    || lower.contains("how")
                    || lower.contains("to")
                    || lower.contains("method");
                if has_proc { 0.3 } else { 0.0 }
            }
            _ => 0.0,
        }
    }

    pub fn rank_sentences(
        sentences: &[String],
        query: &str,
        mode: ReasoningMode,
    ) -> Vec<(String, f64)> {
        let query_terms = Self::extract_terms(query);

        let mut scored: Vec<(String, f64)> = sentences
            .iter()
            .map(|s| (s.clone(), Self::score_sentence(s, &query_terms, mode)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
    }

    pub fn select_top_sentences(
        sentences: &[String],
        query: &str,
        mode: ReasoningMode,
        top_k: usize,
    ) -> Vec<String> {
        let scored = Self::rank_sentences(sentences, query, mode);
        scored.into_iter().take(top_k).map(|(s, _)| s).collect()
    }
}
