// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Framing Topic Derivation
//!
//! Derives the best topic phrase to pass to `ResponseFramer::frame` based on
//! the prompt's reasoning mode.

use crate::cognition::thought::classifier::ReasoningMode;
use crate::constants::cli_tram_response::{AUX_VERBS, CAUSAL_PREFIXES, PREDICATE_STARTERS};

/// Derives the best topic string to pass to [`crate::cognition::response_framer::ResponseFramer::frame`]
/// for a given reasoning mode.
///
/// For `Causal` mode a richer predicate fragment produces a more natural opening.
/// For example, `"why does rust not have a garbage collector?"` frames as
/// `"Rust does not have a garbage collector"` rather than just `"Rust"`.
///
/// # Time Complexity
///
/// O(|prompt|)
#[cfg(feature = "cli")]
pub fn derive_framing_topic(prompt: &str, mode: ReasoningMode, topic: &str) -> String {
    if !matches!(mode, ReasoningMode::Causal) {
        return topic.to_string();
    }
    let lower = prompt.to_lowercase();
    for &(prefix, aux) in CAUSAL_PREFIXES {
        if lower.starts_with(prefix) {
            let rest = prompt[prefix.len()..]
                .trim()
                .trim_end_matches('?')
                .trim_end_matches('.')
                .trim();
            if rest.len() > 2 {
                let words: Vec<&str> = rest.split_whitespace().collect();
                let first_word_lower = words
                    .first()
                    .map(|w| w.to_ascii_lowercase())
                    .unwrap_or_default();
                let needs_aux_insertion = !aux.is_empty()
                    && !AUX_VERBS.contains(&first_word_lower.as_str())
                    && words.len() >= 2;
                let reconstructed = if needs_aux_insertion {
                    let mut subject_words: Vec<&str> = Vec::new();
                    let mut rest_words: Vec<&str> = Vec::new();
                    let mut past_subject = false;
                    for &w in &words {
                        if !past_subject
                            && PREDICATE_STARTERS.contains(&w.to_ascii_lowercase().as_str())
                        {
                            past_subject = true;
                        }
                        if past_subject {
                            rest_words.push(w);
                        } else {
                            subject_words.push(w);
                        }
                    }
                    if subject_words.is_empty() {
                        rest.to_string()
                    } else {
                        let subject = subject_words.join(" ");
                        let predicate = rest_words.join(" ");
                        if predicate.is_empty() {
                            format!("{} {}", subject, aux)
                        } else {
                            format!("{} {} {}", subject, aux, predicate)
                        }
                    }
                } else {
                    rest.to_string()
                };
                let mut chars = reconstructed.chars();
                let first_up: String = chars
                    .next()
                    .map(|c| c.to_uppercase().collect::<String>())
                    .unwrap_or_default();
                return format!("{}{}", first_up, chars.as_str());
            }
        }
    }
    topic.to_string()
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
