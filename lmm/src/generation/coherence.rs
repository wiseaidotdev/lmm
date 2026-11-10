// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Novel Prose Generation: CoherenceScorer
//!
//! Reorders sentences to maximise topical flow by choosing the ordering with the
//! highest sum of adjacent-sentence token overlap.
//!
//! This is a lightweight proxy for semantic coherence: sentences that share
//! many content words are likely to be topically related and should appear
//! close together.
//!
//! ## Algorithm
//!
//! For small sentence sets (≤ 8 sentences) the scorer uses a greedy nearest-
//! neighbour strategy:
//!
//! 1. Start with the sentence that has the highest total overlap with all others
//!    (the "hub" sentence).
//! 2. Repeatedly append the unvisited sentence with the highest overlap to the
//!    last-appended sentence.
//!
//! For larger sets (> 8 sentences), the input order is preserved to keep
//! runtime bounded.

use std::collections::HashSet;

/// Scores sentence ordering for maximum topical flow.
///
/// # Examples
///
/// ```rust
/// use lmm::generation::coherence::CoherenceScorer;
///
/// let sentences = vec![
///     "Rust uses ownership to manage memory.".to_string(),
///     "Memory safety is a core property of Rust.".to_string(),
///     "Python is dynamically typed and interpreted.".to_string(),
/// ];
/// let scorer = CoherenceScorer;
/// let ordered = scorer.reorder(&sentences);
/// assert_eq!(ordered.len(), 3);
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct CoherenceScorer;

impl CoherenceScorer {
    /// Reorders `sentences` for maximum adjacent-sentence token overlap.
    ///
    /// Uses a greedy nearest-neighbour strategy for ≤ 12 sentences; preserves
    /// input order for larger sets.
    ///
    /// # Arguments
    ///
    /// * `sentences` - Slice of sentences to reorder.
    ///
    /// # Returns
    ///
    /// A new `Vec<String>` with sentences in the reordered order.
    ///
    /// # Time Complexity
    ///
    /// O(n² × |sentence|) for n ≤ 12; O(n) otherwise.
    ///
    /// # Space Complexity
    ///
    /// O(n × |tokens_per_sentence|)
    pub fn reorder(&self, sentences: &[String]) -> Vec<String> {
        let n = sentences.len();
        if n <= 1 || n > 12 {
            return sentences.to_vec();
        }

        let token_sets: Vec<HashSet<String>> =
            sentences.iter().map(|s| tokenise_sentence(s)).collect();

        let mut used = vec![false; n];
        let mut result = Vec::with_capacity(n);

        let start = (0..n)
            .max_by_key(|&i| {
                let total: usize = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| overlap(&token_sets[i], &token_sets[j]))
                    .sum();
                total
            })
            .unwrap_or(0);

        result.push(sentences[start].clone());
        used[start] = true;
        let mut last_idx = start;

        for _ in 1..n {
            let last_tokens = &token_sets[last_idx];
            let next = (0..n)
                .filter(|&i| !used[i])
                .max_by_key(|&i| overlap(last_tokens, &token_sets[i]));

            if let Some(idx) = next {
                result.push(sentences[idx].clone());
                used[idx] = true;
                last_idx = idx;
            }
        }

        result
    }
}

/// Tokenises a sentence into a `HashSet` of lowercase content words (≥ 3 chars).
///
/// # Time Complexity
///
/// O(|sentence|)
fn tokenise_sentence(sentence: &str) -> HashSet<String> {
    sentence
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter_map(|w| {
            let w = w.to_ascii_lowercase();
            if w.len() >= 3 { Some(w) } else { None }
        })
        .collect()
}

/// Returns the number of tokens in common between two sets.
///
/// # Time Complexity
///
/// O(min(|a|, |b|))
fn overlap(a: &HashSet<String>, b: &HashSet<String>) -> usize {
    a.iter().filter(|t| b.contains(*t)).count()
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
