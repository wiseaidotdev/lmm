// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Retrieval Stage
//!
//! Queries the [`KnowledgeIndex`] for the best answer to the prompt, applying
//! mode-specific strategies (analogical, evaluative, deep-scrape) and a
//! multi-filter quality pass on deep-scraped paragraphs.

#[cfg(feature = "cli")]
use {
    crate::{
        app::tram::rewards::synthesize_evaluative_points,
        cognition::{
            ReasoningMode,
            knowledge::KnowledgeIndex,
            response_framer::ResponseFramer,
            task_decomposer::{CreativeOutputKind, TaskDecomposer},
        },
        constants::app::{EVALUATIVE_KEYWORDS, VS_SPLITTER_PRIMARY, VS_SPLITTER_SECONDARY},
        constants::tram_answer::{
            ADVANTAGE_KEYWORDS, HISTORICAL_MARKERS, HISTORY_KEYWORDS, MIN_FILTERED_PARAGRAPHS,
        },
        constants::tram_scraper::{
            ARTICLE_GAP_PATTERNS, CITATION_SEMICOLON_AND_GAP_PATTERN,
            CITATION_SEMICOLON_GAP_PATTERN, CITATION_SEMICOLON_WAS_GAP_PATTERN, CODE_INDICATORS,
            CODE_KEYWORDS, DISAMBIGUATION_MARKERS, EMPTY_GAP_PATTERN, INCOMPLETE_COMMON_PATTERN,
            INCOMPLETE_PARENTHETICAL_PATTERNS, INCOMPLETE_QUOTE_EXCEPTION,
            INCOMPLETE_QUOTE_PATTERN, INCOMPLETE_THROUGH_MODULES_PATTERN,
            INCOMPLETE_THROUGH_PATTERN, MAX_DEEP_PARAGRAPHS_TO_SHOW, MIN_DEEP_PARAGRAPH_WORDS,
            MONTH_PATTERNS, SENTENCE_END_INCOMPLETE, STANDALONE_PUNCTUATION, TABLE_INDICATORS,
        },
        tram::split_sentences_with_punct,
    },
    tracing::info,
};

/// Data produced by the retrieval stage.
#[cfg(feature = "cli")]
pub struct RetrievalStageOutput {
    /// KB chunk texts gathered for expert-inference rule extraction.
    pub kb_chunks: Vec<String>,
    /// Best combined answer from the KB, or `None` when no quality answer exists.
    pub combined_answer: Option<String>,
}

/// Applies quality filters to a list of deep-scraped paragraph strings.
///
/// Paragraphs are rejected when they are too short, contain citation artefacts,
/// look like code, table rows, standalone punctuation, incomplete sentences,
/// historical date mentions, or junk markup patterns.
///
/// # Arguments
///
/// * `paragraphs` - Raw paragraph strings from the deep-scrape source.
///
/// # Returns
///
/// A filtered list of up to [`MAX_DEEP_PARAGRAPHS_TO_SHOW`] quality paragraphs.
#[cfg(feature = "cli")]
fn filter_deep_paragraphs(paragraphs: Vec<String>) -> Vec<String> {
    paragraphs
        .into_iter()
        .filter(|s| {
            let words = s.split_whitespace().count();
            let lower = s.to_ascii_lowercase();
            if words < MIN_DEEP_PARAGRAPH_WORDS {
                return false;
            }
            if lower.contains(EMPTY_GAP_PATTERN) {
                return false;
            }
            if lower.contains(CITATION_SEMICOLON_GAP_PATTERN)
                || lower.contains(CITATION_SEMICOLON_AND_GAP_PATTERN)
                || lower.contains(CITATION_SEMICOLON_WAS_GAP_PATTERN)
            {
                return false;
            }
            if lower
                .split_whitespace()
                .any(|w| w.ends_with(';') && w.len() < 5)
            {
                return false;
            }
            if lower.contains(INCOMPLETE_THROUGH_PATTERN)
                || lower.contains(INCOMPLETE_THROUGH_MODULES_PATTERN)
            {
                return false;
            }
            if lower.starts_with(INCOMPLETE_COMMON_PATTERN) {
                return false;
            }
            for pattern in ARTICLE_GAP_PATTERNS {
                if lower.contains(pattern) {
                    return false;
                }
            }
            for keyword in CODE_KEYWORDS {
                if lower.contains(keyword) {
                    return false;
                }
            }
            for indicator in TABLE_INDICATORS {
                if s.contains(indicator) {
                    return false;
                }
            }
            for indicator in CODE_INDICATORS {
                if lower.contains(indicator) {
                    return false;
                }
            }
            for punct in STANDALONE_PUNCTUATION {
                if s.trim().starts_with(*punct) {
                    return false;
                }
            }
            if lower.contains(INCOMPLETE_QUOTE_PATTERN)
                && !lower.contains(INCOMPLETE_QUOTE_EXCEPTION)
            {
                return false;
            }
            for month in MONTH_PATTERNS {
                if lower.starts_with(month) {
                    return false;
                }
            }
            for pattern in INCOMPLETE_PARENTHETICAL_PATTERNS {
                if lower.contains(pattern) {
                    return false;
                }
            }
            for ending in SENTENCE_END_INCOMPLETE {
                if s.ends_with(ending) {
                    return false;
                }
            }
            true
        })
        .take(MAX_DEEP_PARAGRAPHS_TO_SHOW)
        .collect()
}

/// Strips Wikipedia-style disambiguation sentences from a paragraph of text.
///
/// Sentences containing any [`DISAMBIGUATION_MARKERS`] substring (case-insensitive)
/// are removed.
///
/// When all sentences are filtered the original text is returned unchanged.
///
/// # Arguments
///
/// * `text` - Paragraph text from the deep-scraped corpus.
///
/// # Returns
///
/// Filtered text, or the original when all sentences matched.
#[cfg(feature = "cli")]
fn strip_disambiguation_sentences(text: &str) -> String {
    let sentences: Vec<&str> = text.split(". ").collect();
    let filtered: Vec<&str> = sentences
        .iter()
        .copied()
        .filter(|sentence| {
            let lower = sentence.to_ascii_lowercase();
            !DISAMBIGUATION_MARKERS
                .iter()
                .any(|&marker| lower.contains(marker))
        })
        .collect();
    if filtered.is_empty() {
        text.to_string()
    } else {
        filtered.join(". ")
    }
}

/// Strips historical sentences (containing a year digit + a historical marker)
/// from an IA abstract unless the prompt explicitly asks for history.
///
/// # Arguments
///
/// * `text`              - Source text to filter.
/// * `prompt_asks_history` - When `true` the text is returned unmodified.
///
/// # Returns
///
/// Filtered text, or the original text when all sentences were rejected.
#[cfg(feature = "cli")]
fn strip_historical_sentences(text: &str, prompt_asks_history: bool) -> String {
    if prompt_asks_history {
        return text.to_string();
    }
    let split_result = split_sentences_with_punct(text);
    let filtered: Vec<String> = split_result
        .into_iter()
        .filter(|s| {
            let lower = s.to_ascii_lowercase();
            let has_year = lower.chars().any(|c| c.is_ascii_digit());
            let has_historical_marker = HISTORICAL_MARKERS
                .iter()
                .any(|&marker| lower.contains(marker));
            !(has_year && has_historical_marker)
        })
        .collect();
    if filtered.is_empty() {
        text.to_string()
    } else {
        filtered.join(" ")
    }
}

/// Configuration for the retrieval stage.
#[cfg(feature = "cli")]
pub struct RetrievalConfig<'a> {
    /// Fully populated knowledge index.
    pub index: &'a KnowledgeIndex,
    /// Original user prompt.
    pub prompt: &'a str,
    /// Extracted topic phrase.
    pub topic: &'a str,
    /// Classified reasoning mode.
    pub prompt_mode: ReasoningMode,
    /// Per-run seed for top-k jitter.
    pub run_seed: u64,
    /// Whether the ATE/ThinkLoop was active.
    pub think: bool,
    /// Whether deep-scrape enrichment is active.
    pub deep: bool,
    /// Derived KB query string.
    pub kb_query: &'a str,
}

/// Runs the retrieval stage.
///
/// Selects and applies the correct retrieval strategy (analogical, evaluative,
/// evaluative+deep, or generic) and returns the best answer and chunk list.
///
/// # Returns
///
/// A [`RetrievalStageOutput`] with chunks and the best answer.
///
/// # Time Complexity
///
/// O(|index| × top_k)
#[cfg(feature = "cli")]
pub fn run_retrieval_stage(cfg: RetrievalConfig<'_>) -> RetrievalStageOutput {
    let RetrievalConfig {
        index,
        prompt,
        topic,
        prompt_mode,
        run_seed,
        think,
        deep,
        kb_query,
    } = cfg;
    let top_k = 6 + (run_seed % 5) as usize;

    let prompt_lower = prompt.to_ascii_lowercase();
    let is_evaluative = EVALUATIVE_KEYWORDS
        .iter()
        .any(|&keyword| prompt_lower.contains(keyword));

    if matches!(prompt_mode, ReasoningMode::Analogical) {
        return run_analogical_retrieval(index, topic, kb_query, top_k);
    }

    if is_evaluative {
        return run_evaluative_retrieval(EvaluativeRetrievalConfig {
            index,
            prompt,
            topic,
            kb_query,
            prompt_mode,
            top_k,
            think,
            deep,
            run_seed,
        });
    }

    let mut chunks: Vec<String> = index
        .query(kb_query, top_k)
        .into_iter()
        .map(|c| c.text.clone())
        .collect();
    let answer = index
        .answer(kb_query, top_k)
        .or_else(|| index.answer(topic, top_k));

    if TaskDecomposer::is_creative_task(prompt)
        && matches!(
            TaskDecomposer::detect_creative_kind(prompt),
            CreativeOutputKind::Code { .. }
        )
    {
        let code_chunks: Vec<String> = index
            .query_code_blocks()
            .into_iter()
            .map(|c| c.text.clone())
            .collect();
        if !code_chunks.is_empty() {
            info!(
                "  🔍 Retrieval: appending {} code-block chunk(s) to kb_chunks",
                code_chunks.len()
            );
            chunks.extend(code_chunks);
        }
    }

    RetrievalStageOutput {
        kb_chunks: chunks,
        combined_answer: answer,
    }
}

/// Retrieval path for analogical (A vs B) prompts.
#[cfg(feature = "cli")]
fn run_analogical_retrieval(
    index: &KnowledgeIndex,
    topic: &str,
    kb_query: &str,
    top_k: usize,
) -> RetrievalStageOutput {
    let topic_lower = topic.to_ascii_lowercase();
    let mut chunks = Vec::new();
    let vs_splitter = if topic_lower.contains(VS_SPLITTER_PRIMARY) {
        VS_SPLITTER_PRIMARY
    } else if topic_lower.contains(VS_SPLITTER_SECONDARY) {
        VS_SPLITTER_SECONDARY
    } else {
        ""
    };

    let combined_answer = if !vs_splitter.is_empty() {
        if let Some(pos) = topic_lower.find(vs_splitter) {
            let entity_a = topic[..pos].trim();
            let entity_b = topic[pos + vs_splitter.len()..].trim();
            let entity_top_k = top_k.max(3);
            info!(
                "  Analogical retrieval: entity_a='{}', entity_b='{}'",
                entity_a, entity_b
            );
            chunks.extend(
                index
                    .query(entity_a, entity_top_k)
                    .into_iter()
                    .map(|c| c.text.clone()),
            );
            chunks.extend(
                index
                    .query(entity_b, entity_top_k)
                    .into_iter()
                    .map(|c| c.text.clone()),
            );
            let ans_a = index.answer(entity_a, entity_top_k);
            let ans_b = index.answer(entity_b, entity_top_k);
            match (ans_a, ans_b) {
                (Some(a), Some(b)) => Some(format!("{} {}", a, b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        } else {
            chunks.extend(
                index
                    .query(kb_query, top_k)
                    .into_iter()
                    .map(|c| c.text.clone()),
            );
            index.answer(kb_query, top_k)
        }
    } else {
        chunks.extend(
            index
                .query(kb_query, top_k)
                .into_iter()
                .map(|c| c.text.clone()),
        );
        index.answer(kb_query, top_k)
    };

    RetrievalStageOutput {
        kb_chunks: chunks,
        combined_answer,
    }
}

/// Configuration for the evaluative retrieval path (private).
#[cfg(feature = "cli")]
struct EvaluativeRetrievalConfig<'a> {
    index: &'a KnowledgeIndex,
    prompt: &'a str,
    topic: &'a str,
    kb_query: &'a str,
    prompt_mode: ReasoningMode,
    top_k: usize,
    think: bool,
    deep: bool,
    run_seed: u64,
}

/// Retrieval path for evaluative prompts (advantages, benefits, pros/cons).
#[cfg(feature = "cli")]
fn run_evaluative_retrieval(cfg: EvaluativeRetrievalConfig<'_>) -> RetrievalStageOutput {
    let EvaluativeRetrievalConfig {
        index,
        prompt,
        topic,
        kb_query,
        prompt_mode,
        top_k,
        think,
        deep,
        run_seed,
    } = cfg;
    let prompt_lower = prompt.to_ascii_lowercase();
    let mut chunks = Vec::new();
    chunks.extend(
        index
            .query(kb_query, top_k)
            .into_iter()
            .map(|c| c.text.clone()),
    );
    chunks.extend(
        index
            .query(topic, top_k)
            .into_iter()
            .map(|c| c.text.clone()),
    );
    let eval_queries = vec![
        format!("advantages of {}", topic),
        format!("benefits of {}", topic),
        format!("pros and cons of {}", topic),
    ];
    for eq in &eval_queries {
        chunks.extend(index.query(eq, top_k).into_iter().map(|c| c.text.clone()));
    }

    if deep {
        info!("  🔍 Deep mode: querying IA content excluding deep-scraped source");
        let ia_answer = index
            .answer_excluding_source(&eval_queries[0], top_k, "web-deep-scrape")
            .or_else(|| index.answer_excluding_source(topic, top_k, "web-deep-scrape"));

        let all_deep_chunks: Vec<String> = index
            .query_source("web-deep-scrape", top_k * 5)
            .into_iter()
            .map(|c| c.text.clone())
            .collect();
        info!(
            "  🔍 Deep mode: query_source returned {} chunks",
            all_deep_chunks.len()
        );
        chunks.extend(all_deep_chunks.clone());

        let prompt_asks_history = HISTORY_KEYWORDS
            .iter()
            .any(|&keyword| prompt_lower.contains(keyword));

        let filtered_deep: Vec<String> = filter_deep_paragraphs(all_deep_chunks)
            .into_iter()
            .map(|p| strip_disambiguation_sentences(&p))
            .map(|p| strip_historical_sentences(&p, prompt_asks_history))
            .filter(|p| !p.trim().is_empty())
            .collect();

        let answer = if think {
            if let Some(ia) = ia_answer {
                let filtered_ia = strip_historical_sentences(&ia, prompt_asks_history);
                let synthesized_ia = if ADVANTAGE_KEYWORDS
                    .iter()
                    .any(|&keyword| prompt_lower.contains(keyword))
                {
                    synthesize_evaluative_points(&filtered_ia)
                } else {
                    filtered_ia
                };
                let prefixed_ia = if ADVANTAGE_KEYWORDS
                    .iter()
                    .any(|&keyword| prompt_lower.contains(keyword))
                {
                    ResponseFramer::frame(prompt_mode, topic, prompt, &synthesized_ia, run_seed)
                } else {
                    synthesized_ia
                };
                if filtered_deep.len() >= MIN_FILTERED_PARAGRAPHS {
                    Some(format!("{}\n\n{}", prefixed_ia, filtered_deep.join("\n\n")))
                } else {
                    Some(prefixed_ia)
                }
            } else if filtered_deep.len() >= MIN_FILTERED_PARAGRAPHS {
                Some(filtered_deep.join("\n\n"))
            } else {
                index.answer(topic, top_k)
            }
        } else {
            if let Some(ia) = ia_answer {
                if filtered_deep.len() >= MIN_FILTERED_PARAGRAPHS {
                    Some(format!("{}\n\n{}", ia, filtered_deep.join("\n\n")))
                } else {
                    info!(
                        "  🔍 Deep mode: insufficient quality paragraphs ({}), using IA only",
                        filtered_deep.len()
                    );
                    let synthesized = if ADVANTAGE_KEYWORDS
                        .iter()
                        .any(|&kw| prompt_lower.contains(kw))
                    {
                        synthesize_evaluative_points(&ia)
                    } else {
                        ia
                    };
                    Some(synthesized)
                }
            } else if filtered_deep.len() >= MIN_FILTERED_PARAGRAPHS {
                Some(filtered_deep.join("\n\n"))
            } else {
                info!(
                    "  🔍 Deep mode: insufficient quality paragraphs, falling back to regular query"
                );
                index
                    .answer(&eval_queries[0], top_k)
                    .or_else(|| index.answer(topic, top_k))
            }
        };
        return RetrievalStageOutput {
            kb_chunks: chunks,
            combined_answer: answer,
        };
    }

    let answer = index
        .answer(&eval_queries[0], top_k)
        .or_else(|| index.answer(topic, top_k))
        .map(|ia| {
            let prompt_lower = prompt.to_ascii_lowercase();
            if ADVANTAGE_KEYWORDS
                .iter()
                .any(|&kw| prompt_lower.contains(kw))
            {
                synthesize_evaluative_points(&ia)
            } else {
                ia
            }
        });
    RetrievalStageOutput {
        kb_chunks: chunks,
        combined_answer: answer,
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
