// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Symbolic Response Builder
//!
//! Builds a topic-aware fallback response when the knowledge base contains
//! insufficient content to answer the user's prompt.

use crate::constants::cli_tram_response::{
    MAX_SUMMARY_SENTENCES, MIN_COLD_CORPUS_WORDS, MIN_SUMMARY_SENTENCES, MIN_SUMMARY_WORDS,
    SUMMARY_TOP_N, SYMBOLIC_NO_EXPLANATION_TEMPLATE, SYMBOLIC_NO_INFO_TEMPLATE,
    SYMBOLIC_TOPIC_PLACEHOLDER,
};
use crate::text::TextSummarizer;
use crate::tram::extract_topic;

/// Builds a topic-aware symbolic response when no KB answer is available.
///
/// First attempts to summarise the ThinkLoop cold-memory corpus. If that
/// corpus is too thin, returns a graceful "no information available" message
/// grounded in the topic extracted from the prompt, rather than producing
/// mathematical word-salad from `ParagraphGenerator`.
///
/// # Arguments
///
/// * `prompt`       - Original user prompt (used as topic seed).
/// * `cold_corpus`  - Concatenated ThinkLoop cold-store content (may be empty).
///
/// # Time Complexity
///
/// O(|cold_corpus| + |prompt|)
#[cfg(feature = "cli")]
pub fn build_symbolic_response(prompt: &str, cold_corpus: &str) -> String {
    let prompt_lower = prompt.to_ascii_lowercase();
    let is_causal = prompt_lower.starts_with("why ")
        || prompt_lower.starts_with("how come")
        || prompt_lower.contains(" cause")
        || prompt_lower.contains(" reason");

    if cold_corpus.split_whitespace().count() >= MIN_COLD_CORPUS_WORDS {
        let summarizer =
            TextSummarizer::new(MIN_SUMMARY_SENTENCES, MAX_SUMMARY_SENTENCES, SUMMARY_TOP_N);
        if let Ok(sentences) = summarizer.summarize_with_query(cold_corpus, prompt) {
            let joined = sentences.join(" ");
            if joined.split_whitespace().count() >= MIN_SUMMARY_WORDS {
                return joined;
            }
        }
    }

    let topic = extract_topic(prompt);
    let topic_display = if topic.is_empty() {
        prompt
    } else {
        topic.as_str()
    };

    let template = if is_causal {
        SYMBOLIC_NO_EXPLANATION_TEMPLATE
    } else {
        SYMBOLIC_NO_INFO_TEMPLATE
    };
    template.replace(SYMBOLIC_TOPIC_PLACEHOLDER, topic_display)
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
