// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Response Assembly
//!
//! CLI-specific functions for assembling, filtering, and framing TRAM responses.

use crate::cognition::thought::classifier::ReasoningMode;
use crate::generation::{
    dedup::GlobalDeduplicator, filter::SentenceFilter, shaper::ParagraphShaper,
};
use crate::text::TextSummarizer;
use crate::tram::{extract_topic, split_sentences_with_punct, strip_discourse_connector};
use std::cmp::Reverse;
use std::collections::HashSet;

const MIN_QUALITY_WORD_COUNT: usize = 5;
const MIN_SENTENCE_WORD_COUNT: usize = 2;
const MIN_AVG_WORDS_PER_SENTENCE: usize = 2;
const MIN_VERB_COUNT: usize = 1;
const MIN_COLD_CORPUS_WORDS: usize = 20;
const MIN_SUMMARY_WORDS: usize = 10;
const MIN_SUMMARY_SENTENCES: usize = 5;
const MAX_SUMMARY_SENTENCES: usize = 8;
const SUMMARY_TOP_N: usize = 3;
const MIN_TOPIC_WORD_LENGTH: usize = 4;
const MAX_PREDICATE_WORDS: usize = 7;
const MIN_PARAGRAPH_SENTENCES: usize = 2;
const MAX_PARAGRAPH_SENTENCES: usize = 2;
const MIN_SENTENCE_LENGTH: usize = 5;
const MAX_SENTENCE_LENGTH: usize = 40;
const MAX_PARAGRAGH_LENGTH: usize = 180;
const DEDUP_THRESHOLD: f64 = 0.65;

const VERB_PATTERNS: [&str; 21] = [
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " provides ",
    " enables ",
    " uses ",
    " includes ",
    " supports ",
    " allows ",
    " makes ",
    " creates ",
    " consists ",
    " contains ",
    " designed ",
    " built ",
    " formed ",
    " released ",
    " developed ",
];

const ADDITIONAL_VERB_PATTERNS: [&str; 8] = [
    " developed ",
    " began ",
    " defines ",
    " describe ",
    " reason ",
    " causes ",
    " result ",
    " reason",
];

pub const CAUSAL_OPENERS: &[&str] = &[
    "why does ",
    "why do ",
    "why is ",
    "why are ",
    "why was ",
    "why would ",
    "why should ",
    "why can't ",
    "why cannot ",
    "why doesn't ",
    "why don't ",
    "how come ",
];

pub const VS_SPLITTERS: &[&str] = &[" vs ", " versus "];

const SITE_NAMES: &[&str] = &[
    "geeksforgeeks",
    "stackoverflow",
    "reddit",
    "medium",
    "github",
    "wikipedia",
    "quora",
    "hackernews",
    "dev.to",
    "towardsdatascience",
    "freecodecamp",
    "programiz",
    "tutorialspoint",
];

const DANGLING_PREPOSITIONS: &[&str] = &[
    "in", "at", "of", "to", "for", "on", "by", "with", "from", "into", "onto", "upon", "about",
    "over", "under", "between", "among", "through", "within", "without", "during", "and", "or",
    "but", "the", "a", "an",
];

const STOP_WORDS: [&str; 48] = [
    "what",
    "does",
    "how",
    "the",
    "and",
    "for",
    "with",
    "about",
    "install",
    "make",
    "build",
    "create",
    "setup",
    "configure",
    "learn",
    "understand",
    "explain",
    "describe",
    "find",
    "show",
    "tell",
    "give",
    "help",
    "work",
    "works",
    "working",
    "used",
    "using",
    "uses",
    "need",
    "needs",
    "want",
    "that",
    "this",
    "into",
    "from",
    "have",
    "been",
    "will",
    "would",
    "could",
    "should",
    "when",
    "where",
    "which",
    "who",
    "why",
    "also",
];

const ANALOGICAL_VS_SPLITTERS: &[&str] = &[" vs ", " versus ", " compared to "];

const CONNECTORS: &[&str] = &[
    "On the other hand,",
    "By contrast,",
    "Meanwhile,",
    "In comparison,",
    "Conversely,",
];

const CAUSAL_PREFIXES: &[(&str, &str)] = &[
    ("why does ", "does"),
    ("why do ", "do"),
    ("why is ", "is"),
    ("why are ", "are"),
    ("why was ", "was"),
    ("why were ", "were"),
    ("why would ", "would"),
    ("why should ", "should"),
    ("why can't ", "cannot"),
    ("why cannot ", "cannot"),
    ("why doesn't ", "does not"),
    ("why don't ", "do not"),
    ("how come ", ""),
];

const AUX_VERBS: [&str; 23] = [
    "is", "are", "was", "were", "has", "have", "had", "does", "do", "did", "will", "would",
    "could", "should", "can", "cannot", "doesn't", "don't", "can't", "won't", "wouldn't",
    "couldn't", "not",
];

const PREDICATE_STARTERS: &[&str] = &[
    "not", "no", "never", "have", "has", "had", "make", "makes", "work", "works", "allow",
    "allows", "prevent", "prevents", "require", "requires", "cause", "causes", "use", "uses",
    "need", "needs",
];

/// Checks whether a KB answer meets basic quality thresholds.
///
/// Answers below 5 words, with poor sentence structure, or without sufficient
/// verbs are rejected to avoid low-quality responses.
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
    let is_evaluative = prompt_lower.contains("advantage")
        || prompt_lower.contains("benefit")
        || prompt_lower.contains("pros")
        || prompt_lower.contains("disadvantage")
        || prompt_lower.contains("inconvenience")
        || prompt_lower.contains("limitation")
        || prompt_lower.contains("drawback")
        || prompt_lower.contains("weakness")
        || prompt_lower.contains("cons");
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

    if is_evaluative {
        format!(
            "The {} of {} are as follows. No detailed information was found for \"{}\". \
             This may be because the topic is very specific, the web query returned no results, \
             or the knowledge base does not yet contain content on this subject. \
             Try rephrasing your query or adding more context.",
            if prompt_lower.contains("disadvantage") || prompt_lower.contains("inconvenience") {
                "inconveniences"
            } else {
                "advantages"
            },
            topic_display,
            topic_display
        )
    } else if is_causal {
        format!(
            "No detailed explanation was found for \"{}\". \
             This may be because the topic is very specific, the web query returned no results, \
             or the knowledge base does not yet contain content on this subject. \
             Try rephrasing your query or adding more context.",
            topic_display
        )
    } else {
        format!(
            "No detailed information was found for \"{}\". \
             This may be because the topic is very specific, the web query returned no results, \
             or the knowledge base does not yet contain content on this subject. \
             Try rephrasing your query or adding more context.",
            topic_display
        )
    }
}

/// Derives the most relevant KB query string for a given reasoning mode and prompt.
///
/// For `Evaluative` mode, focuses on advantages, features, or drawbacks depending
/// on the prompt's intent. For `Causal` mode, strips the question prefix so the
/// predicate phrase is used for TF-IDF retrieval. For `Analogical` mode, generates
/// queries for both entities in a comparison. Otherwise returns the original
/// prompt unchanged.
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
                .unwrap_or("");
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

/// Assembles paragraphs from retrieved text, prioritizing sentences that mention
/// the topic entities while filtering out metadata and low-quality content.
///
/// For comparison topics (e.g. "rust vs python"), sentences mentioning both
/// entities are scored higher to ensure both sides of the comparison appear
/// in the output.
///
/// # Time Complexity
///
/// O(|text| * |keywords|)
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
                let a_first = a.split_whitespace().next().unwrap_or("").to_string();
                let b_first = b.split_whitespace().next().unwrap_or("").to_string();
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
                .unwrap_or("")
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
            let first = s.split_whitespace().next().unwrap_or("");
            if first.len() < 4 {
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
            if has_year && has_historical_marker && !prompt_asks_history {
                return false;
            }
            true
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

/// Derives the best topic string to pass to [`ResponseFramer::frame`] for a
/// given reasoning mode.
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
