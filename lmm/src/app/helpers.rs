// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Application Helper Utilities

#[cfg(any(feature = "cli", feature = "net"))]
use crate::tram::extract_topic;
#[cfg(feature = "cli")]
use crate::{cli::util, error::Result as LmmResult, stochastic::StochasticEnhancer};

/// Loads a text source from either a file path or a provided string.
///
/// # Arguments
///
/// * `input` - File path string.
/// * `text`  - Fallback inline text when no file is provided.
///
/// # Returns
///
/// The loaded or provided text as a [`LmmResult<String>`].
#[cfg(feature = "cli")]
pub fn load_source(input: &str, text: String) -> LmmResult<String> {
    util::load_source(input, text)
}

/// Optionally applies stochastic synonym substitution to the given text.
///
/// # Arguments
///
/// * `text`        - The source text to enhance.
/// * `stochastic`  - When `false`, returns the text unchanged.
/// * `probability` - Per-word substitution probability when enhancement is active.
///
/// # Returns
///
/// The (potentially enhanced) text string.
#[cfg(feature = "cli")]
pub fn maybe_enhance(text: &str, stochastic: bool, probability: f64) -> String {
    if stochastic {
        StochasticEnhancer::new(probability).enhance(text)
    } else {
        text.to_string()
    }
}

/// Extracts the topic name from a query for answer-prefix generation.
///
/// Handles patterns like `"what are the advantages of X?"` and
/// `"what are the benefits of X?"`, returning the capitalised topic name.
///
/// # Arguments
///
/// * `prompt` - The raw user prompt string.
///
/// # Returns
///
/// A capitalised topic string.
#[cfg(feature = "net")]
pub fn extract_topic_name(prompt: &str) -> String {
    let prompt_lower = prompt.to_ascii_lowercase();

    if prompt_lower.contains("advantages of")
        && let Some(pos) = prompt_lower.find("advantages of")
    {
        let topic = prompt[pos + "advantages of".len()..].trim();
        let topic = topic.trim_end_matches('?').trim_end_matches('.');
        return capitalize_first_letter(topic);
    }

    if prompt_lower.contains("benefits of")
        && let Some(pos) = prompt_lower.find("benefits of")
    {
        let topic = prompt[pos + "benefits of".len()..].trim();
        let topic = topic.trim_end_matches('?').trim_end_matches('.');
        return capitalize_first_letter(topic);
    }

    let topic = extract_topic(prompt);
    capitalize_first_letter(&topic)
}

/// Capitalises the first letter of a string slice.
///
/// # Arguments
///
/// * `s` - Input string slice.
///
/// # Returns
///
/// A new `String` with the first character upper-cased.
#[cfg(feature = "net")]
#[allow(unused)]
pub fn capitalize_first_letter(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
