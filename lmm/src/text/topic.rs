// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Topic extraction utilities for summarizing responses into specific topics.

use crate::constants::text_topic::{
    DEFAULT_TOPIC_FALLBACK, ELLIPSIS_PATTERN, FALSE_TOPIC_WORDS, FIRST_SENTENCES_LIMIT,
    GENERAL_ASPECTS, GENERIC_NOUNS, GENERIC_TOPIC_PHRASES, MIN_CORE_WORD_LENGTH,
    MIN_ELLIPSIS_COUNT, MIN_TOPIC_WORD_LENGTH, MIN_WORD_COUNT_FOR_BAD_RESPONSE, PERSON_TITLES,
    PROGRAMMING_ASPECTS, STOPWORDS, TECH_COMPOUNDS, TECH_SIGNALS, TOPIC_CANDIDATE_LIMIT,
};

fn is_tech_domain(domain: &str) -> bool {
    let lo = domain.to_lowercase();
    TECH_SIGNALS.iter().any(|&s| lo.contains(s))
}

fn domain_key(domain: &str) -> &str {
    domain
        .split_whitespace()
        .find(|w| {
            let lo = w.to_lowercase();
            w.len() >= 4
                && !STOPWORDS.contains(&lo.as_str())
                && !FALSE_TOPIC_WORDS.contains(&lo.as_str())
        })
        .unwrap_or_else(|| {
            domain
                .split_whitespace()
                .next()
                .unwrap_or(DEFAULT_TOPIC_FALLBACK)
        })
}

fn is_person_name_context(words: &[&str], i: usize) -> bool {
    for item in words.iter().take(i).skip(i.saturating_sub(3)) {
        let w = item.to_lowercase();
        let w_clean = w.trim_matches(|c: char| !c.is_alphabetic());
        if PERSON_TITLES.contains(&w_clean) {
            return true;
        }
    }
    if i > 0 {
        let prev = words[i - 1].to_lowercase();
        let prev_clean = prev.trim_matches(|c: char| !c.is_alphabetic());
        if prev_clean == "by" {
            return true;
        }
    }
    false
}

fn is_bad_response(text: &str) -> bool {
    if GENERIC_TOPIC_PHRASES
        .iter()
        .any(|&phrase| text.contains(phrase))
    {
        return true;
    }
    if text.matches(ELLIPSIS_PATTERN).count() >= MIN_ELLIPSIS_COUNT {
        return true;
    }
    if text.split_whitespace().count() < MIN_WORD_COUNT_FOR_BAD_RESPONSE {
        return true;
    }
    false
}

/// Generate an aspect-based topic for a given domain and turn.
pub fn aspect_topic(domain: &str, turn: u64) -> String {
    let key = domain_key(domain);
    if is_tech_domain(domain) {
        let aspect = PROGRAMMING_ASPECTS[(turn as usize) % PROGRAMMING_ASPECTS.len()];
        format!("{} {}", key, aspect)
    } else {
        let aspect = GENERAL_ASPECTS[(turn as usize) % GENERAL_ASPECTS.len()];
        format!("{} {}", key, aspect)
    }
}

/// Summarize a response text into a specific topic for the given domain.
///
/// This function extracts the most relevant topic from the response by:
/// 1. Checking for bad/generic responses and falling back to aspect_topic
/// 2. Extracting capitalized words that are not stopwords or false topics
/// 3. Checking for technical compound phrases in tech domains
/// 4. Falling back to content word extraction
/// 5. Final fallback to aspect_topic
pub fn summarize_to_topic(text: &str, domain: &str, current_topic: &str, turn: u64) -> String {
    let clean = text.trim();
    if clean.is_empty() {
        return aspect_topic(domain, turn + 1);
    }

    if is_bad_response(clean) {
        return aspect_topic(domain, turn + 1);
    }

    let first_sentences: Vec<&str> = clean
        .split(['.', '!', '?'])
        .take(FIRST_SENTENCES_LIMIT)
        .collect();

    for sentence in &first_sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        for (i, &word) in words.iter().enumerate() {
            if i == 0 || word.len() < MIN_TOPIC_WORD_LENGTH {
                continue;
            }
            if word.contains('-') || word.contains('_') {
                continue;
            }
            let core: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            if core.len() < MIN_TOPIC_WORD_LENGTH {
                continue;
            }
            let mut chars = core.chars();
            let first_char = match chars.next() {
                Some(c) => c,
                None => continue,
            };
            if !first_char.is_uppercase() {
                continue;
            }
            let core_lower = core.to_lowercase();
            if STOPWORDS.contains(&core_lower.as_str()) {
                continue;
            }
            if FALSE_TOPIC_WORDS.contains(&core_lower.as_str()) {
                continue;
            }
            if is_person_name_context(&words, i) {
                continue;
            }

            let rest_all_lower = core.chars().skip(1).all(|c| c.is_lowercase());

            if rest_all_lower {
                let domain_words: Vec<&str> = domain.split_whitespace().collect();
                let pivot_in_domain = domain_words
                    .iter()
                    .any(|&dw| dw.to_lowercase() == core_lower);

                if pivot_in_domain {
                    if let Some(pivot_pos) = domain_words
                        .iter()
                        .position(|&dw| dw.to_lowercase() == core_lower)
                        && let Some(&next_dw) = domain_words.get(pivot_pos + 1)
                        && next_dw.len() >= 4
                        && !STOPWORDS.contains(&next_dw.to_lowercase().as_str())
                    {
                        let candidate = format!("{} {}", core, next_dw);
                        if !candidate.eq_ignore_ascii_case(current_topic) {
                            return candidate;
                        }
                    }
                } else {
                    if !core.eq_ignore_ascii_case(current_topic) {
                        return core;
                    }
                }
            } else {
                if !core.eq_ignore_ascii_case(current_topic) {
                    return core;
                }
            }
        }
    }

    if is_tech_domain(domain) {
        let clean_lower = clean.to_lowercase();
        let key = domain_key(domain);
        for &compound in TECH_COMPOUNDS {
            if clean_lower.contains(&compound.to_lowercase()) {
                let candidate = format!("{} {}", key, compound);
                if !candidate.eq_ignore_ascii_case(current_topic)
                    && !compound.eq_ignore_ascii_case(current_topic)
                {
                    return candidate;
                }
            }
        }
    }

    for word in clean.split_whitespace() {
        if word.len() < MIN_CORE_WORD_LENGTH || word.contains('-') {
            continue;
        }
        let core: String = word.chars().filter(|c| c.is_alphabetic()).collect();
        if core.len() < MIN_CORE_WORD_LENGTH {
            continue;
        }
        let mut ch = core.chars();
        let up = ch.next().map(|c| c.is_uppercase()).unwrap_or(false);
        let lo_rest = ch.all(|c| c.is_lowercase());
        if !up || !lo_rest {
            continue;
        }
        let lo = core.to_lowercase();
        if STOPWORDS.contains(&lo.as_str()) || FALSE_TOPIC_WORDS.contains(&lo.as_str()) {
            continue;
        }
        if lo == domain_key(domain).to_lowercase() {
            continue;
        }
        if !core.eq_ignore_ascii_case(current_topic) {
            return core;
        }
    }

    let first_sentence = clean.split(['.', '!', '?']).next().unwrap_or(clean);

    let content_words: Vec<&str> = first_sentence
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| {
            let lo = w.to_lowercase();
            w.len() >= 7
                && !STOPWORDS.contains(&lo.as_str())
                && !FALSE_TOPIC_WORDS.contains(&lo.as_str())
                && !GENERIC_NOUNS.contains(&lo.as_str())
                && !w.contains('-')
                && w.chars().any(|c| c.is_alphabetic())
        })
        .take(TOPIC_CANDIDATE_LIMIT)
        .collect();

    if content_words.len() >= 2 {
        let candidate = content_words.join(" ");
        if !candidate.eq_ignore_ascii_case(current_topic) {
            return candidate;
        }
    }

    aspect_topic(domain, turn + 1)
}
