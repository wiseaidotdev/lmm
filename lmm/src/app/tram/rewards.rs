// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Reward and Synthesis Utilities
//!
//! Episode-level quality scoring and evaluative-point extraction used by the
//! assembly stage to close the reinforcement-learning loop.

#[cfg(feature = "cli")]
use crate::{
    constants::{
        tram_pipeline::{
            REWARD_KB_COVERAGE_WEIGHT, REWARD_MIN_ANSWER_WORDS, REWARD_NOVEL_CONTENT_BONUS,
            REWARD_PROMPT_RELEVANCE_WEIGHT,
        },
        tram_response::{
            EVALUATIVE_EXCLUSION_PATTERNS, EVALUATIVE_SIGNAL_WORDS, MIN_EVALUATIVE_SENTENCES,
        },
    },
    tram::split_sentences_with_punct,
};

/// Computes a quality reward `[0.0, 1.0]` for the generated answer.
///
/// The reward is based on three signals:
/// - **Length adequacy**: answers shorter than [`REWARD_MIN_ANSWER_WORDS`] score 0.
/// - **Prompt relevance**: fraction of prompt keywords present in the answer.
/// - **Novel content bonus**: applied when the answer is not a verbatim
///   substring of the prompt.
///
/// # Arguments
///
/// * `answer` - The final TRAM answer string.
/// * `prompt` - The original user prompt.
///
/// # Returns
///
/// A reward value in `[0.0, 1.0]`.
#[cfg(feature = "cli")]
pub fn compute_response_reward(answer: &str, prompt: &str) -> f64 {
    let word_count = answer.split_whitespace().count();
    if word_count < REWARD_MIN_ANSWER_WORDS {
        return 0.0;
    }

    let prompt_lower = prompt.to_ascii_lowercase();
    let answer_lower = answer.to_ascii_lowercase();
    let prompt_keywords: Vec<&str> = prompt_lower
        .split_whitespace()
        .filter(|w| w.len() >= 4)
        .collect();
    let relevance = if prompt_keywords.is_empty() {
        0.5
    } else {
        let matched = prompt_keywords
            .iter()
            .filter(|&&kw| answer_lower.contains(kw))
            .count();
        matched as f64 / prompt_keywords.len() as f64
    };

    let novel_bonus = if !prompt_lower.contains(&answer_lower[..answer_lower.len().min(40)]) {
        REWARD_NOVEL_CONTENT_BONUS
    } else {
        0.0
    };

    (relevance * REWARD_PROMPT_RELEVANCE_WEIGHT + REWARD_KB_COVERAGE_WEIGHT * 0.5 + novel_bonus)
        .min(1.0)
}

/// Strips only historical/attribution sentences from `ia_text`, without
/// applying signal-word gating.
///
/// Used as a fallback when the full evaluative filter retains fewer than
/// [`crate::constants::tram_response::MIN_EVALUATIVE_SENTENCES`] sentences.
///
/// # Time Complexity
///
/// O(|ia_text| × |EVALUATIVE_EXCLUSION_PATTERNS|)
#[cfg(feature = "cli")]
fn strip_historical_only(ia_text: &str) -> String {
    let sentences = split_sentences_with_punct(ia_text);
    let kept: Vec<String> = sentences
        .into_iter()
        .filter(|s| {
            let lower = s.to_ascii_lowercase();
            !EVALUATIVE_EXCLUSION_PATTERNS
                .iter()
                .any(|&pat| lower.contains(pat))
        })
        .collect();
    if kept.is_empty() {
        ia_text.to_string()
    } else {
        kept.join(" ")
    }
}

/// Extracts evaluative points (features, capabilities) from a definitional
/// IA abstract when the pipeline lacks richer KB content.
///
/// Each sentence in `ia_text` is tested against [`crate::constants::tram_response::EVALUATIVE_SIGNAL_WORDS`].
/// Sentences that contain at least one signal word **and** do not match any
/// entry in [`crate::constants::tram_response::EVALUATIVE_EXCLUSION_PATTERNS`] are kept.
///
/// The exclusion patterns drop historical, attribution, and definitional
/// sentences (e.g. "was created in 2006", "sponsored by") that contain signal
/// words incidentally.  When fewer than [`crate::constants::tram_response::MIN_EVALUATIVE_SENTENCES`]
/// sentences pass both filters, the function falls back to `strip_historical_only`
/// so that at least some useful content is returned. When no sentence passes
/// that fallback either, the full `ia_text` is returned unchanged.
///
/// # Arguments
///
/// * `ia_text` - The Internet Archive abstract or similar definitional text.
///
/// # Returns
///
/// A filtered string of evaluative sentences, or `ia_text` verbatim if none qualify.
///
/// # Time Complexity
///
/// O(|ia_text| × (|EVALUATIVE_SIGNAL_WORDS| + |EVALUATIVE_EXCLUSION_PATTERNS|))
#[cfg(feature = "cli")]
pub fn synthesize_evaluative_points(ia_text: &str) -> String {
    let sentences = split_sentences_with_punct(ia_text);
    let points: Vec<String> = sentences
        .into_iter()
        .filter(|s| {
            let lower = s.to_ascii_lowercase();
            let has_signal = EVALUATIVE_SIGNAL_WORDS
                .iter()
                .any(|&signal| lower.contains(signal));
            let is_excluded = EVALUATIVE_EXCLUSION_PATTERNS
                .iter()
                .any(|&pat| lower.contains(pat));
            has_signal && !is_excluded
        })
        .collect();
    if points.len() >= MIN_EVALUATIVE_SENTENCES {
        return points.join(" ");
    }
    strip_historical_only(ia_text)
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
