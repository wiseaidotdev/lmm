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

use phf::phf_set;

const FILLER_PREFIXES: &[&str] = &[
    "what is a ",
    "what is an ",
    "what is the ",
    "what is ",
    "what are the ",
    "what are ",
    "what was ",
    "what were ",
    "how does a ",
    "how does an ",
    "how does ",
    "how do ",
    "how is ",
    "how are ",
    "how to ",
    "explain ",
    "describe ",
    "define ",
    "tell me about ",
    "what's ",
    "whats ",
    "who is ",
    "who was ",
    "why is a ",
    "why is an ",
    "why is the ",
    "why is ",
    "why are ",
    "why does ",
    "why do ",
    "where is ",
    "when is ",
    "give me information about ",
    "give me info on ",
    "write a ",
    "write an ",
    "write ",
    "generate a ",
    "generate an ",
    "create a ",
    "create an ",
    "implement a ",
    "implement an ",
    "code a ",
    "build a ",
    "show me how to ",
    "show me a ",
    "show me ",
    "give me a ",
    "give me an ",
    "compare ",
];

const MODE_KEYWORD_PREFIXES: &[&str] = &[
    "advantages of ",
    "advantage of ",
    "benefits of ",
    "benefit of ",
    "disadvantages of ",
    "disadvantage of ",
    "pros and cons of ",
    "pros of ",
    "cons of ",
    "drawbacks of ",
    "drawback of ",
    "strengths of ",
    "strength of ",
    "weaknesses of ",
    "weakness of ",
    "limitations of ",
    "limitation of ",
    "inconveniences of ",
    "inconvenience of ",
    "upsides of ",
    "downsides of ",
    "merits of ",
    "merit of ",
    "demerits of ",
    "demerit of ",
    "causes of ",
    "cause of ",
    "reason for ",
    "reasons for ",
    "explanation for ",
    "effect of ",
    "effects of ",
    "definition of ",
    "meaning of ",
    "concept of ",
    "examples of ",
    "overview of ",
    "history of ",
    "installation of ",
    "setup of ",
    "usage of ",
    "use of ",
    "uses of ",
    "comparison of ",
    "difference between ",
    "differences between ",
    "similarities between ",
    "relationship between ",
    "why is ",
    "why are ",
    "why does ",
    "why do ",
];

const PREDICATE_WORDS: &[&str] = &[
    "not", "no", "never", "have", "has", "had", "is", "are", "was", "were", "do", "does", "did",
    "will", "would", "could", "should", "can", "cannot", "work", "works", "make", "makes", "use",
    "uses", "need", "needs", "allow", "allows", "prevent", "prevents", "require", "requires",
    "cause", "causes",
];

const TRAILING_VERBS: &[&str] = &[
    " work",
    " works",
    " function",
    " functions",
    " operate",
    " operates",
    " run",
    " runs",
    "do",
    "does",
    " mean",
    " means",
    " help",
    "helps",
    "differ",
    "differs",
];

const CONNECTOR_PREFIXES: &[&str] = &[
    "Furthermore, ",
    "In addition, ",
    "Notably, ",
    "Moreover, ",
    "As a result, ",
    "In practice, ",
    "Interestingly, ",
    "Consequently, ",
    "Building on this, ",
    "To elaborate, ",
    "More specifically, ",
    "On the technical side, ",
    "From a practical standpoint, ",
    "In this context, ",
    "Also, ",
    "However, ",
    "Therefore, ",
    "Thus, ",
    "Hence, ",
    "More precisely, ",
    "This is because ",
    "To be specific, ",
    "It is well established that ",
    "At its core, ",
    "Put simply, ",
    "Broadly speaking, ",
    "In technical terms, ",
    "By design, ",
    "In essence, ",
    "Based on observed knowledge ",
    "To elaborate on this, ",
];

const ABBREVIATIONS: phf::Set<&'static str> = phf_set! {
    "e.g", "i.e", "vs", "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "no", "fig", "cf", "etc", "approx", "dept", "est", "govt", "inc",
    "ltd", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
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
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim_end_matches('!')
            .trim()
    } else {
        prompt
            .trim()
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim_end_matches('!')
            .trim()
    };
    let raw_lower = raw_after_filler.to_ascii_lowercase();

    if raw_lower.starts_with("why is ") || raw_lower.starts_with("why are ") {
        let words: Vec<&str> = raw_after_filler.split_whitespace().collect();
        if words.len() >= 3 {
            let entity = words[2];
            if !entity.is_empty() {
                return entity.to_string();
            }
        }
    }

    if raw_lower.contains(" vs ") {
        return raw_after_filler.to_string();
    }

    if raw_lower.contains(" and ") {
        let parts: Vec<&str> = raw_after_filler.split(" and ").collect();
        if parts.len() == 2 {
            let entity_a = parts[0].trim();
            let entity_b = parts[1].trim();
            return format!("{} vs {}", entity_a, entity_b);
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
    for article in &["a ", "an ", "the "] {
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
    if words.len() < 3 {
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
                    .unwrap_or("")
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
            if s.split_whitespace().count() >= 8 {
                out.push(s);
            }
            cur.clear();
        }
    }

    let tail = cur.trim().to_string();
    if tail.split_whitespace().count() >= 8 {
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
