// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Query Derivation
//!
//! Functions for extracting topics, stripping common patterns, and deriving
//! knowledge base queries from user prompts.

use crate::constants::generation_filter::DEFAULT_EMPTY_WORD;
use crate::constants::task_decomposer::TASK_CREATIVE_VERBS;
use crate::constants::tram_query::{
    ABBREVIATIONS, AND_CONNECTOR, ARTICLE_A, ARTICLE_AN, ARTICLE_PREFIX_A, ARTICLE_PREFIX_AN,
    ARTICLE_THE, CONNECTOR_PREFIXES, CREATIVE_FORM_WORDS, CREATIVE_SUBJECT_CONNECTORS,
    ENTITY_INDEX_WHY_PATTERN, FILLER_PREFIXES, MIN_WORDS_FOR_SENTENCE,
    MIN_WORDS_FOR_TRIM_PREDICATE, MIN_WORDS_FOR_WHY_PATTERN, MODE_KEYWORD_PREFIXES,
    PREDICATE_WORDS, PUNCTUATION_EXCLAMATION, PUNCTUATION_PERIOD, PUNCTUATION_QUESTION,
    SIGNIFICANT_NOUN_FALLBACK_TAKE, SIGNIFICANT_NOUN_FIRST_WORD_MIN_LEN,
    SIGNIFICANT_NOUN_MIN_WORDS, SIGNIFICANT_NOUN_STOP_WORDS, SUBJECT_FORM_SUFFIXES,
    SUBJECT_LANGUAGE_PREFIXES, SUBJECT_MODIFIER_PREFIXES, TRAILING_VERBS, VS_CONNECTOR,
    WHY_ARE_PREFIX, WHY_IS_PREFIX,
};

/// Extracts a clean topic keyword phrase from a natural-language prompt by
/// stripping common question prefixes and filler words.
///
/// The DuckDuckGo Instant Answer API returns rich Wikipedia summaries for
/// concise noun phrases ("Rust programming language") but nothing for
/// question forms ("what is rust programming"). This function normalises the
/// prompt before the IA query while keeping the full text for ThinkLoop.
///
/// # Time Complexity
///
/// O(|prompt|)
pub fn extract_topic(prompt: &str) -> String {
    let lower = prompt.to_lowercase();
    let after_filler: Option<&str> = FILLER_PREFIXES
        .iter()
        .find(|&&prefix| lower.starts_with(prefix))
        .map(|&prefix| &prompt[prefix.len()..]);
    let raw_after_filler = if let Some(r) = after_filler {
        r.trim()
            .trim_end_matches(PUNCTUATION_QUESTION)
            .trim_end_matches(PUNCTUATION_PERIOD)
            .trim_end_matches(PUNCTUATION_EXCLAMATION)
            .trim()
    } else {
        prompt
            .trim()
            .trim_end_matches(PUNCTUATION_QUESTION)
            .trim_end_matches(PUNCTUATION_PERIOD)
            .trim_end_matches(PUNCTUATION_EXCLAMATION)
            .trim()
    };
    let raw_lower = raw_after_filler.to_ascii_lowercase();

    if raw_lower.starts_with(WHY_IS_PREFIX) || raw_lower.starts_with(WHY_ARE_PREFIX) {
        let words: Vec<&str> = raw_after_filler.split_whitespace().collect();
        if words.len() >= MIN_WORDS_FOR_WHY_PATTERN {
            let entity = words[ENTITY_INDEX_WHY_PATTERN];
            if !entity.is_empty() {
                return entity.to_string();
            }
        }
    }

    if raw_lower.contains(VS_CONNECTOR) {
        return raw_after_filler.to_string();
    }

    if raw_lower.contains(AND_CONNECTOR) {
        let parts: Vec<&str> = raw_after_filler.split(AND_CONNECTOR).collect();
        if parts.len() == 2 {
            let entity_a = parts[0].trim();
            let entity_b = parts[1].trim();
            return format!("{}{}{}", entity_a, VS_CONNECTOR, entity_b);
        }
    }
    let after_mode = MODE_KEYWORD_PREFIXES
        .iter()
        .find(|&pfx| raw_lower.starts_with(pfx))
        .map(|&pfx| raw_after_filler[pfx.len()..].trim())
        .unwrap_or(raw_after_filler);
    let stripped = strip_leading_articles(after_mode);
    if after_filler.is_some() {
        strip_trailing_verb(trim_predicate(&stripped))
    } else {
        stripped
    }
}

/// Strips a leading article ("a", "an", "the") from a noun phrase.
///
/// # Time Complexity
///
/// O(|s|)
pub fn strip_leading_articles(s: &str) -> String {
    let lower = s.to_lowercase();
    for article in &[ARTICLE_A, ARTICLE_AN, ARTICLE_THE] {
        if lower.starts_with(article) {
            return s[article.len()..].trim().to_string();
        }
    }
    s.to_string()
}

/// Trims a topic string at the first predicate word, keeping only the subject noun phrase.
///
/// When a question like `"why does Rust not have a garbage collector"` is stripped
/// of its question prefix, the remainder `"Rust not have a garbage collector"` still
/// contains the predicate.  This function truncates at the first auxiliary, negation,
/// or main-verb word so only `"Rust"` is returned, a far better search term.
///
/// Only applies when the input has 3 or more words to avoid over-truncating short topics.
///
/// # Time Complexity
///
/// O(|s|)
pub fn trim_predicate(s: &str) -> String {
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < MIN_WORDS_FOR_TRIM_PREDICATE {
        return s.to_string();
    }
    let cut = words
        .iter()
        .skip(1)
        .position(|w| PREDICATE_WORDS.contains(&w.to_ascii_lowercase().as_str()))
        .map(|i| i + 1)
        .unwrap_or(words.len());
    words[..cut].join(" ")
}

/// Strips a trailing common verb from a topic string extracted from a question.
///
/// For example, `"how does tokio work"` → `extract_topic` strips `"how does "`
/// yielding `"tokio work"`. This function further strips `" work"` to give
/// just `"tokio"`, which produces cleaner DuckDuckGo Instant Answer hits.
///
/// # Time Complexity
///
/// O(|s|)
pub fn strip_trailing_verb(s: String) -> String {
    let lower = s.to_lowercase();
    for verb in TRAILING_VERBS {
        if lower.ends_with(verb) {
            return s[..s.len() - verb.len()].trim().to_string();
        }
    }
    s
}

/// Splits text into complete sentences with terminal punctuation attached.
///
/// Sentences below 8 words are silently discarded.
///
/// Handles common English abbreviations (`e.g.`, `i.e.`, `vs.`, `Dr.`,
/// `Mr.`, `Mrs.`, `No.`, `Fig.`, `cf.`) to prevent spurious sentence splits
/// at their trailing periods.
///
/// # Time Complexity
///
/// O(|text|)
pub fn split_sentences_with_punct(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();

    for (i, &ch) in chars.iter().enumerate() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let next_char = chars.get(i + 1).copied();
            let next_lower = next_char.map(|c| c.is_ascii_lowercase()).unwrap_or(false);
            let next_space = next_char.map(|c| c == ' ').unwrap_or(false);
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = next_char.map(|c| c.is_ascii_digit()).unwrap_or(false);

            if ch == '.' && (next_lower || (prev_digit && next_digit)) {
                continue;
            }

            if ch == '.' && next_space {
                let word_before: String = cur
                    .trim_end_matches('.')
                    .split_whitespace()
                    .last()
                    .unwrap_or(DEFAULT_EMPTY_WORD)
                    .chars()
                    .filter(|c| c.is_ascii_alphabetic() || *c == '.')
                    .collect::<String>()
                    .to_ascii_lowercase();
                let stripped = word_before.trim_end_matches('.');
                if ABBREVIATIONS.contains(stripped) {
                    continue;
                }
            }

            let next_upper = next_char.map(|c| c.is_ascii_uppercase()).unwrap_or(false);
            if ch == '.'
                && !next_upper
                && !next_char.map(|c| c == '\n').unwrap_or(false)
                && next_char.is_some()
                && !next_space
            {
                continue;
            }

            let s = cur.trim().to_string();
            if s.split_whitespace().count() >= MIN_WORDS_FOR_SENTENCE {
                out.push(s);
            }
            cur.clear();
        }
    }

    let tail = cur.trim().to_string();
    if tail.split_whitespace().count() >= MIN_WORDS_FOR_SENTENCE {
        if !tail.ends_with('.') && !tail.ends_with('!') && !tail.ends_with('?') {
            out.push(format!("{}.", tail));
        } else {
            out.push(tail);
        }
    }

    let _ = len;
    out
}

/// Strips a leading discourse connector phrase from a sentence.
///
/// Connectors like `"Furthermore, "`, `"In addition, "`, `"As a result, "` are
/// prepended by [`ProseBuilder`] and [`ParagraphShaper`] when building
/// non-first paragraphs.  When sentences are later re-shuffled through
/// [`assemble_paragraphs`], these stale connectors cause double-stacking
/// (e.g. `"To elaborate, Furthermore, rust supports..."`).  This function
/// removes them so the shaper can apply fresh connectors at the right positions.
///
/// # Time Complexity
///
/// O(|sentence|)
pub fn strip_discourse_connector(sentence: String) -> String {
    for prefix in CONNECTOR_PREFIXES {
        if let Some(rest) = sentence.strip_prefix(prefix) {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                let mut chars = trimmed.chars();
                let first_up: String = chars
                    .next()
                    .map(|c| c.to_uppercase().collect::<String>())
                    .unwrap_or_default();
                return first_up + chars.as_str();
            }
        }
    }
    sentence
}

/// Extracts the domain subject from a creative prompt by stripping both the
/// creative verb and the form-word (poem, program, story, ...) so that only the
/// research topic remains.
///
/// # Examples
///
/// * `"write a poem about life"` → `"life"`
/// * `"write a simple python program computing factorial of a number"` → `"factorial"`
/// * `"generate a fibonacci function in rust"` → `"fibonacci"`
///
/// When the prompt does not look like a creative task the function falls back
/// to `extract_topic`.
///
/// # Time Complexity
///
/// O(|prompt|)
pub fn extract_creative_subject(prompt: &str) -> String {
    let lower = prompt.to_ascii_lowercase();
    let trimmed = lower
        .trim()
        .trim_end_matches(PUNCTUATION_QUESTION)
        .trim_end_matches(PUNCTUATION_PERIOD)
        .trim_end_matches(PUNCTUATION_EXCLAMATION)
        .trim();

    let after_verb: &str = TASK_CREATIVE_VERBS
        .iter()
        .find(|&&v| trimmed.starts_with(v))
        .map(|&v| &trimmed[v.len()..])
        .unwrap_or(trimmed);

    let after_article = after_verb
        .trim_start_matches(ARTICLE_PREFIX_AN)
        .trim_start_matches(ARTICLE_PREFIX_A);

    let after_form: &str = CREATIVE_FORM_WORDS
        .iter()
        .find(|&&fw| {
            let fw_no_article = fw
                .trim_start_matches(ARTICLE_PREFIX_AN)
                .trim_start_matches(ARTICLE_PREFIX_A);
            after_article.starts_with(fw_no_article)
        })
        .map(|&fw| {
            let fw_no_article = fw
                .trim_start_matches(ARTICLE_PREFIX_AN)
                .trim_start_matches(ARTICLE_PREFIX_A);
            &after_article[fw_no_article.len()..]
        })
        .unwrap_or(after_article);

    let after_connector: &str = CREATIVE_SUBJECT_CONNECTORS
        .iter()
        .find(|&&con| after_form.starts_with(con))
        .map(|&con| &after_form[con.len()..])
        .unwrap_or(after_form);

    let subject_raw = after_connector
        .trim()
        .trim_end_matches(PUNCTUATION_QUESTION)
        .trim_end_matches(PUNCTUATION_PERIOD)
        .trim_end_matches(PUNCTUATION_EXCLAMATION)
        .trim();

    let cleaned = strip_subject_noise(subject_raw);
    let subject = trim_to_first_significant_noun(&cleaned);

    if subject.is_empty() || subject == trimmed {
        extract_topic(prompt)
    } else {
        subject
    }
}

/// Strips language-name prefixes, generic adjective modifiers, and form-noun
/// suffixes from a raw subject phrase, leaving the core domain concept.
///
/// Examples:
/// - `"simple python hello world program"` → `"hello world"`
/// - `"derivative of a number"` → `"derivative of a number"` (unchanged)
/// - `"python factorial function"` → `"factorial"`
///
/// Stripping is applied in order: language prefix → modifier prefixes
/// (iteratively) → form-noun suffix.
///
/// # Time Complexity
///
/// O(|s| × (|prefixes| + |suffixes|))
fn strip_subject_noise(s: &str) -> String {
    let mut result = s.to_string();
    let mut changed = true;
    while changed {
        changed = false;
        for &prefix in SUBJECT_LANGUAGE_PREFIXES {
            if result.starts_with(prefix) {
                result = result[prefix.len()..].to_string();
                changed = true;
                break;
            }
        }
        for &prefix in SUBJECT_MODIFIER_PREFIXES {
            if result.starts_with(prefix) {
                result = result[prefix.len()..].to_string();
                changed = true;
                break;
            }
        }
    }
    for &suffix in SUBJECT_FORM_SUFFIXES {
        if result.ends_with(suffix) {
            result = result[..result.len() - suffix.len()].to_string();
            break;
        }
    }
    result.trim().to_string()
}

/// Trims a subject phrase down to its first significant noun word(s).
///
/// For example `"factorial of a number"` → `"factorial"`;
/// `"life"` → `"life"` (unchanged).
///
/// Only applies when the phrase has 3 or more words and the first word is
/// more than 2 characters long, to avoid over-stripping short subjects.
///
/// # Time Complexity
///
/// O(|s|)
fn trim_to_first_significant_noun(s: &str) -> String {
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < SIGNIFICANT_NOUN_MIN_WORDS {
        return s.to_string();
    }
    let first = words[0];
    if first.len() <= SIGNIFICANT_NOUN_FIRST_WORD_MIN_LEN {
        return s.to_string();
    }
    let cut = words
        .iter()
        .skip(1)
        .position(|w| SIGNIFICANT_NOUN_STOP_WORDS.contains(&w.to_ascii_lowercase().as_str()))
        .map(|i| i + 1)
        .unwrap_or(words.len().min(SIGNIFICANT_NOUN_FALLBACK_TAKE));
    words[..cut].join(" ")
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
