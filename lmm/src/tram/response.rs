// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Response Quality and Assembly
//!
//! Functions for filtering response quality, checking relevance, framing responses,
//! filtering and deduplicating sentences, restructuring analogical responses, and
//! assembling paragraphs.

use std::collections::HashSet;

const MIN_RESPONSE_LENGTH: usize = 30;
const MAX_REPETITION_THRESHOLD: f64 = 0.4;
const MIN_SENTENCE_LENGTH: usize = 5;
const MIN_SENTENCE_VARIETY: usize = 3;
const MIN_WORDS_FOR_VALID_SUMMARY: usize = 8;
const MIN_SENTENCE_LEN_FOR_VALID_SUMMARY: usize = 3;
const MIN_VERB_COUNT_FOR_VALID_SUMMARY: usize = 1;

const RELEVANCE_KEYWORDS: [&str; 7] = [
    "related",
    "relevant",
    "pertinent",
    "connected",
    "applicable",
    "associated",
    "linked",
];

const IRRELEVANCE_KEYWORDS: [&str; 6] = [
    "unrelated",
    "irrelevant",
    "disconnected",
    "inapplicable",
    "foreign",
    "extraneous",
];

const QUALITY_KEYWORDS: [&str; 8] = [
    "comprehensive",
    "thorough",
    "detailed",
    "complete",
    "accurate",
    "precise",
    "informative",
    "substantial",
];

const POOR_QUALITY_KEYWORDS: [&str; 8] = [
    "brief",
    "incomplete",
    "limited",
    "superficial",
    "shallow",
    "inadequate",
    "fragmented",
    "minimal",
];

const REFUSAL_PATTERNS: [&str; 5] = [
    "i don't know",
    "i cannot",
    "i'm not sure",
    "i am not sure",
    "unknown",
];

const LOW_CONTENT_PATTERNS: [&str; 4] =
    ["title only", "no information", "insufficient", "not enough"];

const RELEVANCE_INTENT_PATTERNS: [&str; 6] = [
    "what is", "explain", "describe", "how does", "why is", "define",
];

const INTENT_KEYWORD_MAP: [(&str, [&str; 5]); 3] = [
    (
        "advantages",
        ["benefit", "pro", "strength", "merit", "gain"],
    ),
    (
        "disadvantages",
        ["drawback", "con", "weakness", "limitation", "flaw"],
    ),
    (
        "comparison",
        ["versus", "vs", "against", "compared", "difference"],
    ),
];

const STRUCTURAL_MARKERS: [&str; 8] = [
    "first",
    "second",
    "third",
    "finally",
    "moreover",
    "furthermore",
    "however",
    "therefore",
];

const ANALOGY_PATTERNS: [&str; 5] = [
    "like a",
    "similar to",
    "analogous to",
    "resembles",
    "comparable to",
];

const COMPARISON_PATTERNS: [&str; 5] = [
    "compared to",
    "in contrast to",
    "unlike",
    "whereas",
    "while",
];

const PARAGRAPH_CONNECTORS: [&str; 10] = [
    "furthermore",
    "in addition",
    "moreover",
    "additionally",
    "besides",
    "also",
    "moreover",
    "consequently",
    "as a result",
    "therefore",
];

const PROSE_CONNECTORS: [&str; 15] = [
    "furthermore",
    "in addition",
    "moreover",
    "additionally",
    "consequently",
    "as a result",
    "therefore",
    "thus",
    "hence",
    "however",
    "nevertheless",
    "in contrast",
    "notably",
    "significantly",
    "interestingly",
];

const SHAPER_CONNECTORS: [&str; 8] = [
    "in practice",
    "more specifically",
    "on the technical side",
    "from a practical standpoint",
    "in this context",
    "to elaborate",
    "to be specific",
    "building on this",
];

/// Checks if a response meets basic quality criteria.
///
/// A quality response must have minimum length, avoid excessive repetition,
/// and contain substantive content beyond refusal patterns.
pub fn is_quality_answer(answer: &str) -> bool {
    if answer.len() < MIN_RESPONSE_LENGTH {
        return false;
    }

    let lower = answer.to_lowercase();
    for pattern in REFUSAL_PATTERNS {
        if lower.contains(pattern) {
            return false;
        }
    }

    for pattern in LOW_CONTENT_PATTERNS {
        if lower.contains(pattern) {
            return false;
        }
    }

    let words: Vec<&str> = answer.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }

    let unique_words: HashSet<&str> = words.iter().cloned().collect();
    let repetition_ratio = 1.0 - (unique_words.len() as f64 / words.len() as f64);
    if repetition_ratio > MAX_REPETITION_THRESHOLD {
        return false;
    }

    true
}

/// Calculates a relevance score for a response against a query.
///
/// Returns a value between 0.0 and 1.0, where higher values indicate greater relevance.
pub fn calculate_relevance_score(response: &str, query: &str) -> f64 {
    let response_lower = response.to_lowercase();
    let query_lower = query.to_lowercase();
    let query_words: HashSet<&str> = query_lower.split_whitespace().collect();
    let response_words: Vec<&str> = response_lower.split_whitespace().collect();

    if query_words.is_empty() || response_words.is_empty() {
        return 0.0;
    }

    let mut keyword_matches = 0;
    for word in &response_words {
        if query_words.contains(word) {
            keyword_matches += 1;
        }
    }

    let match_ratio = keyword_matches as f64 / response_words.len() as f64;

    let mut relevance_bonus = 0.0;
    for keyword in RELEVANCE_KEYWORDS {
        if response_lower.contains(keyword) {
            relevance_bonus += 0.05;
        }
    }

    let mut irrelevance_penalty = 0.0;
    for keyword in IRRELEVANCE_KEYWORDS {
        if response_lower.contains(keyword) {
            irrelevance_penalty += 0.1;
        }
    }

    let mut quality_bonus = 0.0;
    for keyword in QUALITY_KEYWORDS {
        if response_lower.contains(keyword) {
            quality_bonus += 0.03;
        }
    }

    let mut poor_quality_penalty = 0.0;
    for keyword in POOR_QUALITY_KEYWORDS {
        if response_lower.contains(keyword) {
            poor_quality_penalty += 0.05;
        }
    }

    let mut intent_bonus = 0.0;
    for pattern in RELEVANCE_INTENT_PATTERNS {
        if query_lower.contains(pattern) {
            for (intent, keywords) in INTENT_KEYWORD_MAP {
                if query_lower.contains(intent) {
                    for keyword in keywords {
                        if response_lower.contains(keyword) {
                            intent_bonus += 0.02;
                        }
                    }
                }
            }
        }
    }

    let base_score = match_ratio;
    let total_bonus = relevance_bonus + quality_bonus + intent_bonus;
    let total_penalty = irrelevance_penalty + poor_quality_penalty;

    (base_score + total_bonus - total_penalty).clamp(0.0, 1.0)
}

/// Frames a response with appropriate introductory and concluding context.
///
/// Enhances the response with structural markers and discourse connectors
/// to improve readability and coherence.
pub fn frame_response(response: &str, query: &str) -> String {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let query_lower = query.to_lowercase();
    let mut framed = String::new();

    let has_intro = STRUCTURAL_MARKERS
        .iter()
        .any(|m| trimmed.to_lowercase().starts_with(m));

    if !has_intro {
        for pattern in RELEVANCE_INTENT_PATTERNS {
            if query_lower.contains(pattern) {
                framed.push_str("In response to your inquiry, ");
                break;
            }
        }
    }

    framed.push_str(trimmed);

    let has_conclusion = STRUCTURAL_MARKERS
        .iter()
        .any(|m| trimmed.to_lowercase().contains(m));

    if !has_conclusion && trimmed.len() > 100 {
        if !framed.ends_with('.') {
            framed.push('.');
        }
        framed.push_str(" This provides a comprehensive overview of the topic.");
    }

    framed
}

/// Filters sentences based on length and variety criteria.
///
/// Removes sentences that are too short or too similar to existing sentences.
pub fn filter_sentences(sentences: &[String]) -> Vec<String> {
    let mut filtered: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for sentence in sentences {
        let word_count = sentence.split_whitespace().count();
        if word_count < MIN_SENTENCE_LENGTH {
            continue;
        }

        let normalized = sentence
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>();

        if seen.contains(&normalized) {
            continue;
        }

        seen.insert(normalized);
        filtered.push(sentence.clone());
    }

    if filtered.len() < MIN_SENTENCE_VARIETY {
        return sentences.to_vec();
    }

    filtered
}

/// Deduplicates sentences while preserving order.
///
/// Removes duplicate sentences based on normalized content comparison.
pub fn deduplicate_sentences(sentences: &[String]) -> Vec<String> {
    let mut deduped: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for sentence in sentences {
        let normalized = sentence
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>();

        if !seen.contains(&normalized) {
            seen.insert(normalized);
            deduped.push(sentence.clone());
        }
    }

    deduped
}

/// Restructures an analogical response for better coherence.
///
/// Enhances analogical explanations with appropriate discourse markers
/// and structural connectors.
pub fn restructure_analogical_response(response: &str) -> String {
    let lower = response.to_lowercase();
    let mut restructured = response.to_string();

    for pattern in ANALOGY_PATTERNS {
        if lower.contains(pattern) {
            if !restructured.to_lowercase().starts_with("analogous") {
                restructured = format!("Analogous to this, {}", restructured);
            }
            break;
        }
    }

    for pattern in COMPARISON_PATTERNS {
        if lower.contains(pattern) {
            if !lower.contains("however") && !lower.contains("in contrast") {
                restructured.push_str(" However, this comparison reveals important distinctions.");
            }
            break;
        }
    }

    restructured
}

/// Assembles filtered sentences into coherent paragraphs.
///
/// Groups sentences into paragraphs with appropriate discourse connectors
/// and transition phrases.
pub fn assemble_paragraphs(sentences: &[String]) -> String {
    if sentences.is_empty() {
        return String::new();
    }

    let mut paragraphs: Vec<String> = Vec::new();
    let mut current_paragraph: Vec<String> = Vec::new();

    for (i, sentence) in sentences.iter().enumerate() {
        current_paragraph.push(sentence.clone());

        if i % 3 == 2 && i != sentences.len() - 1 {
            let connector = PARAGRAPH_CONNECTORS
                .get(i / 3 % PARAGRAPH_CONNECTORS.len())
                .unwrap_or(&"additionally");
            let joined = current_paragraph.join(" ");
            paragraphs.push(format!("{} {},", joined, connector));
            current_paragraph.clear();
        }
    }

    if !current_paragraph.is_empty() {
        paragraphs.push(current_paragraph.join(" "));
    }

    paragraphs.join(" ")
}

/// Checks if a summary meets minimum quality criteria.
///
/// Evaluates word count, sentence length, and verb presence to determine
/// if a summary is substantive enough for use.
pub fn is_valid_summary(summary: &str) -> bool {
    let word_count = summary.split_whitespace().count();
    if word_count < MIN_WORDS_FOR_VALID_SUMMARY {
        return false;
    }

    let sentences: Vec<&str> = summary
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .collect();

    let valid_sentences = sentences
        .iter()
        .filter(|s| s.split_whitespace().count() >= MIN_SENTENCE_LEN_FOR_VALID_SUMMARY)
        .count();

    if valid_sentences == 0 {
        return false;
    }

    let lower = summary.to_lowercase();
    let verb_count = VERB_PATTERNS.iter().filter(|v| lower.contains(*v)).count();

    verb_count >= MIN_VERB_COUNT_FOR_VALID_SUMMARY
}

const VERB_PATTERNS: [&str; 20] = [
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
];

/// Enhances a response with prose-level discourse connectors.
///
/// Adds appropriate connectors to improve flow and coherence between sentences.
pub fn enhance_with_prose_connectors(response: &str) -> String {
    let sentences: Vec<&str> = response
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .collect();

    if sentences.len() <= 1 {
        return response.to_string();
    }

    let mut enhanced = String::new();
    for (i, sentence) in sentences.iter().enumerate() {
        let trimmed = sentence.trim();
        if trimmed.is_empty() {
            continue;
        }

        if i > 0 {
            let connector = PROSE_CONNECTORS
                .get((i - 1) % PROSE_CONNECTORS.len())
                .unwrap_or(&"furthermore");
            enhanced.push_str(&format!(" {}, {}", connector, to_title_case(trimmed)));
        } else {
            enhanced.push_str(trimmed);
        }
        enhanced.push('.');
    }

    enhanced
}

/// Enhances a response with paragraph-level shapers.
///
/// Adds context-specific connectors to improve paragraph structure.
pub fn enhance_with_shaper_connectors(response: &str) -> String {
    let sentences: Vec<&str> = response
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .collect();

    if sentences.len() <= 2 {
        return response.to_string();
    }

    let mut enhanced = String::new();
    for (i, sentence) in sentences.iter().enumerate() {
        let trimmed = sentence.trim();
        if trimmed.is_empty() {
            continue;
        }

        if i >= 2 {
            let connector = SHAPER_CONNECTORS
                .get((i - 2) % SHAPER_CONNECTORS.len())
                .unwrap_or(&"in practice");
            enhanced.push_str(&format!(" {}, {}", connector, to_title_case(trimmed)));
        } else {
            enhanced.push_str(trimmed);
        }
        enhanced.push('.');
    }

    enhanced
}

fn to_title_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
