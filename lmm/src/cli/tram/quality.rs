// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Answer Quality Checks
//!
//! Functions that test whether a KB answer meets minimum quality and intent
//! thresholds before being accepted by the assembly stage.

use crate::cognition::thought::classifier::ReasoningMode;
use crate::constants::cli_tram_response::{
    ADDITIONAL_VERB_PATTERNS, MIN_AVG_WORDS_PER_SENTENCE, MIN_QUALITY_WORD_COUNT,
    MIN_SENTENCE_WORD_COUNT, MIN_TOPIC_WORD_LENGTH, MIN_VERB_COUNT, VERB_PATTERNS,
};
use std::collections::HashSet;

/// Checks whether a KB answer meets basic quality thresholds.
///
/// Answers below [`MIN_QUALITY_WORD_COUNT`] words, with poor sentence structure,
/// or without sufficient verbs are rejected to avoid low-quality responses.
///
/// # Arguments
///
/// * `answer` - Raw text from `KnowledgeIndex::answer`.
///
/// # Time Complexity
///
/// O(|answer|)
#[cfg(feature = "cli")]
pub fn is_quality_answer(answer: &str) -> bool {
    let word_count = answer.split_whitespace().count();
    if word_count < MIN_QUALITY_WORD_COUNT {
        return false;
    }
    let sentences: Vec<&str> = answer
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.split_whitespace().count() >= MIN_SENTENCE_WORD_COUNT)
        .collect();
    let sentence_count = sentences.len().max(1);
    let avg_words_per_sentence = word_count / sentence_count;

    if avg_words_per_sentence < MIN_AVG_WORDS_PER_SENTENCE {
        return false;
    }

    let verb_count = sentences
        .iter()
        .filter(|s| {
            let lower = s.to_lowercase();
            VERB_PATTERNS.iter().any(|p| lower.contains(p))
                || ADDITIONAL_VERB_PATTERNS.iter().any(|p| lower.contains(p))
        })
        .count();
    verb_count >= MIN_VERB_COUNT
}

/// Checks whether a KB answer is sufficiently relevant to the user's intent.
///
/// For evaluative and causal prompts the Wikipedia-style prose often provides
/// only general/historical facts. When the answer does not contain at least
/// one intent-signal word we fall back to the symbolic response so the framing
/// opener does not dangle over unrelated content.
///
/// For other modes the function always returns `true` (pass-through).
///
/// # Time Complexity
///
/// O(|answer|)
#[cfg(feature = "cli")]
pub fn is_answer_intent_relevant(answer: &str, mode: ReasoningMode, prompt: &str) -> bool {
    let lower = answer.to_ascii_lowercase();
    let prompt_lower = prompt.to_ascii_lowercase();

    let answer_words: HashSet<&str> = lower.split_whitespace().collect();
    let prompt_keywords: Vec<&str> = prompt_lower
        .split_whitespace()
        .filter(|w| w.len() >= MIN_TOPIC_WORD_LENGTH)
        .collect();

    let has_question_keywords = prompt_keywords.iter().any(|&w| answer_words.contains(w));
    let has_topic_keywords = prompt_keywords.iter().any(|&w| lower.contains(w));

    if !has_topic_keywords && !has_question_keywords {
        return false;
    }

    match mode {
        ReasoningMode::Evaluative => has_topic_keywords,
        ReasoningMode::Causal => has_topic_keywords,
        ReasoningMode::Analogical => {
            let topic_words: Vec<&str> = prompt.split_whitespace().collect();
            topic_words
                .iter()
                .any(|&w| lower.contains(w.to_ascii_lowercase().as_str()))
                || has_topic_keywords
        }
        _ => true,
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
