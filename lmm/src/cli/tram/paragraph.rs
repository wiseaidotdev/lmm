// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Paragraph Assembly
//!
//! Functions for assembling and restructuring paragraphs from retrieved text,
//! including the analogical two-entity restructuring pass.

use crate::constants::cli_tram_response::{
    ANALOGICAL_VS_SPLITTERS, CONNECTORS, DANGLING_PREPOSITIONS, DEDUP_THRESHOLD,
    MAX_PARAGRAGH_LENGTH, MAX_PARAGRAPH_SENTENCES, MAX_SENTENCE_LENGTH, MIN_PARAGRAPH_SENTENCES,
    MIN_SENTENCE_LENGTH, MIN_TOPIC_WORD_LENGTH, SITE_NAMES, STOP_WORDS, VS_SPLITTERS,
};
use crate::constants::text_generation::DEFAULT_EMPTY_STRING;
use crate::generation::{
    dedup::GlobalDeduplicator, filter::SentenceFilter, shaper::ParagraphShaper,
};
use crate::tram::{extract_topic, split_sentences_with_punct, strip_discourse_connector};
use std::cmp::Reverse;

/// Assembles paragraphs from retrieved text, prioritising sentences that mention
/// the topic entities while filtering out metadata and low-quality content.
///
/// For comparison topics (e.g. "rust vs python"), sentences mentioning both
/// entities are scored higher to ensure both sides of the comparison appear
/// in the output.
///
/// # Time Complexity
///
/// O(|text| × |keywords|)
#[cfg(feature = "cli")]
pub fn assemble_paragraphs_for(text: &str, run_seed: u64, topic: &str) -> String {
    let topic_cleaned = extract_topic(topic);
    let topic_ref = if topic_cleaned.is_empty() {
        topic
    } else {
        topic_cleaned.as_str()
    };

    let topic_ref_lower = topic_ref.to_ascii_lowercase();
    let topic_words: Vec<String> = {
        let mut words: Vec<String> = Vec::new();
        let mut is_comparison = false;
        for &splitter in VS_SPLITTERS {
            if let Some(pos) = topic_ref_lower.find(splitter) {
                let a = topic_ref[..pos].trim().to_ascii_lowercase();
                let b = topic_ref[pos + splitter.len()..]
                    .trim()
                    .to_ascii_lowercase();
                let a_first = a
                    .split_whitespace()
                    .next()
                    .unwrap_or(DEFAULT_EMPTY_STRING)
                    .to_string();
                let b_first = b
                    .split_whitespace()
                    .next()
                    .unwrap_or(DEFAULT_EMPTY_STRING)
                    .to_string();
                if !a_first.is_empty() {
                    words.push(a_first);
                }
                if !b_first.is_empty() {
                    words.push(b_first);
                }
                is_comparison = true;
                break;
            }
        }
        if !is_comparison {
            let first = topic_ref
                .split_whitespace()
                .find(|w| w.len() > 3 && !STOP_WORDS.contains(w))
                .unwrap_or(DEFAULT_EMPTY_STRING)
                .to_lowercase();
            if !first.is_empty() {
                words.push(first);
            }
        }
        words
    };

    let prompt_lower = topic.to_ascii_lowercase();
    let is_analogical = prompt_lower.contains(" vs ") || prompt_lower.contains(" versus ");
    let query_keywords: Vec<&str> = if is_analogical {
        let mut keywords = Vec::new();
        for splitter in [" vs ", " versus "].iter() {
            if let Some(pos) = prompt_lower.find(*splitter) {
                let entity_a = prompt_lower[..pos].trim();
                let entity_b = prompt_lower[pos + splitter.len()..].trim();
                if entity_a.len() > 3 {
                    keywords.push(entity_a);
                }
                if entity_b.len() > 3 {
                    keywords.push(entity_b);
                }
                break;
            }
        }
        if keywords.is_empty() {
            prompt_lower
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .collect()
        } else {
            keywords
        }
    } else {
        prompt_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect()
    };

    let sentences: Vec<String> = split_sentences_with_punct(text)
        .into_iter()
        .map(strip_discourse_connector)
        .filter(|s| SentenceFilter::is_acceptable(s))
        .filter(|s| {
            let trimmed = s.trim();
            if trimmed.starts_with('-') || trimmed.starts_with('•') || trimmed.starts_with('*') {
                return false;
            }
            let lower = trimmed.to_ascii_lowercase();
            !SITE_NAMES.iter().any(|&site| lower.contains(site))
        })
        .filter(|s| {
            let last = s
                .trim_end_matches(['.', '!', '?'])
                .split_whitespace()
                .last()
                .unwrap_or("")
                .to_ascii_lowercase();
            !DANGLING_PREPOSITIONS.contains(&last.as_str())
        })
        .filter(|s| {
            if is_analogical {
                return true;
            }
            if topic_words.is_empty() {
                return true;
            }
            let first = s.split_whitespace().next().unwrap_or(DEFAULT_EMPTY_STRING);
            if first.len() < MIN_TOPIC_WORD_LENGTH {
                return true;
            }
            let mut chars = first.chars();
            let starts_upper = chars.next().map(|c| c.is_uppercase()).unwrap_or(false);
            let rest_lower = chars.all(|c| c.is_lowercase());
            if !starts_upper || !rest_lower {
                return true;
            }
            let first_lower = first.to_lowercase();
            let s_lower = s.to_lowercase();
            topic_words.iter().any(|tw| {
                first_lower.contains(tw.as_str())
                    || tw.contains(first_lower.as_str())
                    || s_lower.contains(tw.as_str())
            })
        })
        .filter(|s| {
            let lower = s.to_ascii_lowercase();
            let has_year = lower.chars().any(|c| c.is_ascii_digit());
            let has_historical_marker = lower.contains("created")
                || lower.contains("founded")
                || lower.contains("sponsored")
                || lower.contains("release")
                || lower.contains("developer")
                || lower.contains("foundation");
            let prompt_asks_history = prompt_lower.contains("history")
                || prompt_lower.contains("when")
                || prompt_lower.contains("created")
                || prompt_lower.contains("founded");
            !(has_year && has_historical_marker && !prompt_asks_history)
        })
        .collect();

    if sentences.is_empty() {
        return text.to_string();
    }

    fn score_sentence(s: &str, keywords: &[&str]) -> i32 {
        let lower = s.to_ascii_lowercase();
        let mut score = 0i32;
        for kw in keywords {
            if lower.contains(kw) {
                score += 3;
            }
        }
        score
    }

    let mut scored: Vec<(i32, String)> = sentences
        .into_iter()
        .map(|s| (score_sentence(&s, &query_keywords), s.clone()))
        .collect();

    scored.sort_by_key(|a| Reverse(a.0));

    let best_score = scored.first().map(|(sc, _)| *sc).unwrap_or(0);
    let top_tier: Vec<String> = scored
        .iter()
        .filter(|(sc, _)| *sc >= best_score)
        .map(|(_, s)| s.clone())
        .collect();
    let rest: Vec<String> = scored
        .iter()
        .filter(|(sc, _)| *sc < best_score)
        .map(|(_, s)| s.clone())
        .collect();

    let mut top_shuffled = top_tier;
    let top_n = top_shuffled.len();
    let mut seed = run_seed;
    for i in (1..top_n).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        top_shuffled.swap(i, j);
    }

    let mut rest_shuffled = rest;
    let rest_n = rest_shuffled.len();
    for i in (1..rest_n).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        rest_shuffled.swap(i, j);
    }

    top_shuffled.extend(rest_shuffled);
    let sentences = top_shuffled;

    let deduplicator = GlobalDeduplicator::new(DEDUP_THRESHOLD);
    let mut sentences = deduplicator.deduplicate(sentences);

    if sentences.is_empty() {
        return text.to_string();
    }

    if let Some(good_pos) = sentences
        .iter()
        .position(|s| SentenceFilter::is_good_opener(s))
    {
        sentences.swap(0, good_pos);
    }

    let shaper = ParagraphShaper::new(
        MIN_PARAGRAPH_SENTENCES,
        MAX_PARAGRAPH_SENTENCES,
        MAX_PARAGRAGH_LENGTH,
        MIN_SENTENCE_LENGTH,
        MAX_SENTENCE_LENGTH,
    );
    shaper.shape(&sentences, run_seed)
}

/// Restructures an analogical (comparison) response so each entity's sentences
/// appear in a dedicated paragraph separated by a natural-language connector
/// phrase (e.g. `"On the other hand,"`).
///
/// The function extracts the opening sentence (the framed opener such as
/// `"Comparing Rust and Python: the key differences are as follows."`) and
/// then groups the remaining sentences by which comparison entity they discuss.
/// Entity-A sentences form the first paragraph; Entity-B sentences form the
/// second, preceded by a connector. Sentences that mention both entities
/// (or neither) are placed in the first paragraph.
///
/// When only one entity is mentioned throughout, the original `framed` text is
/// returned unchanged to avoid an empty second paragraph.
///
/// # Time Complexity
///
/// O(|framed|)
#[cfg(feature = "cli")]
pub fn restructure_analogical_response(framed: &str, topic: &str, run_seed: u64) -> String {
    let topic_lower = topic.to_ascii_lowercase();
    let mut entity_a = String::new();
    let mut entity_b = String::new();
    for splitter in ANALOGICAL_VS_SPLITTERS {
        if let Some(pos) = topic_lower.find(splitter) {
            entity_a = topic_lower[..pos].trim().to_string();
            entity_b = topic_lower[pos + splitter.len()..].trim().to_string();
            break;
        }
    }
    if entity_a.is_empty() || entity_b.is_empty() {
        return framed.to_string();
    }
    let opener_end = framed.find(". ").map(|i| i + 2).unwrap_or(framed.len());
    let opener = &framed[..opener_end.min(framed.len())];
    let body = framed[opener_end.min(framed.len())..].trim();
    if body.is_empty() {
        return framed.to_string();
    }
    let raw_sentences: Vec<String> = split_sentences_with_punct(body)
        .into_iter()
        .map(strip_discourse_connector)
        .filter(|s| SentenceFilter::is_acceptable(s))
        .collect();
    if raw_sentences.is_empty() {
        return framed.to_string();
    }
    let mut a_sentences: Vec<String> = Vec::new();
    let mut b_sentences: Vec<String> = Vec::new();
    let mut neutral: Vec<String> = Vec::new();
    for s in raw_sentences {
        let sl = s.to_ascii_lowercase();
        let mentions_a = sl.contains(entity_a.as_str());
        let mentions_b = sl.contains(entity_b.as_str());
        match (mentions_a, mentions_b) {
            (true, false) => a_sentences.push(s),
            (false, true) => b_sentences.push(s),
            _ => neutral.push(s),
        }
    }
    if b_sentences.is_empty() {
        return framed.to_string();
    }
    let connector = CONNECTORS[run_seed as usize % CONNECTORS.len()];
    let mut para_a: Vec<String> = neutral;
    para_a.extend(a_sentences);
    let mut parts: Vec<String> = vec![opener.trim().to_string()];
    if !para_a.is_empty() {
        parts.push(para_a.join(" "));
    }
    let b_first = b_sentences[0].clone();
    let b_rest = b_sentences[1..].join(" ");
    let b_para = if b_rest.is_empty() {
        format!("{} {}", connector, b_first)
    } else {
        format!("{} {} {}", connector, b_first, b_rest)
    };
    parts.push(b_para);
    parts.join("\n\n")
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
