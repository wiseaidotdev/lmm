// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM KB Query Derivation
//!
//! Derives the most relevant knowledge-base query string for a given
//! reasoning mode so retrieval is as targeted as possible.

use crate::cognition::thought::classifier::ReasoningMode;
use crate::constants::cli_tram_response::{CAUSAL_OPENERS, MAX_PREDICATE_WORDS, VS_SPLITTERS};
use crate::constants::text_generation::DEFAULT_EMPTY_STRING;

/// Derives the most relevant KB query string for a given reasoning mode and prompt.
///
/// For `Evaluative` mode, focuses on advantages, features, or drawbacks depending
/// on the prompt's intent. For `Causal` mode, strips the question prefix so the
/// predicate phrase is used for TF-IDF retrieval. For `Analogical` mode, generates
/// a merged entity query. Otherwise returns the original prompt unchanged.
///
/// # Time Complexity
///
/// O(|prompt|)
#[cfg(feature = "cli")]
pub fn derive_kb_query(prompt: &str, mode: ReasoningMode, topic: &str) -> String {
    match mode {
        ReasoningMode::Evaluative => {
            let pl = prompt.to_ascii_lowercase();
            let evaluative_word = pl
                .split_whitespace()
                .find(|w| {
                    w.len() > 4
                        && !matches!(
                            *w,
                            "what"
                                | "does"
                                | "have"
                                | "that"
                                | "this"
                                | "with"
                                | "from"
                                | "into"
                                | "are"
                                | "the"
                                | "about"
                        )
                })
                .unwrap_or(DEFAULT_EMPTY_STRING);
            if !evaluative_word.is_empty() {
                format!("{} {}", evaluative_word, topic)
            } else {
                topic.to_string()
            }
        }
        ReasoningMode::Causal => {
            let lower = prompt.to_ascii_lowercase();
            let lower_clean = lower.trim_end_matches('?').trim_end_matches('.').trim();
            let predicate = CAUSAL_OPENERS
                .iter()
                .find_map(|&pfx| lower_clean.strip_prefix(pfx))
                .unwrap_or(lower_clean)
                .trim();
            let words: Vec<&str> = predicate.split_whitespace().collect();
            let truncated = if words.len() > MAX_PREDICATE_WORDS {
                words[..MAX_PREDICATE_WORDS].join(" ")
            } else {
                predicate.to_string()
            };
            if truncated.len() > topic.len() + 2 {
                truncated
            } else {
                prompt.to_string()
            }
        }
        ReasoningMode::Analogical => {
            let topic_lower = topic.to_ascii_lowercase();
            for splitter in VS_SPLITTERS {
                if let Some(pos) = topic_lower.find(splitter) {
                    let entity_a = topic[..pos].trim();
                    let entity_b = topic[pos + splitter.len()..].trim();
                    return format!("{} {}", entity_a, entity_b);
                }
            }
            topic.to_string()
        }
        _ => prompt.to_string(),
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
