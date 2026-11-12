// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Topic extraction utilities for summarizing responses into specific topics.

const TECH_SIGNALS: &[&str] = &[
    "programming",
    "language",
    "algorithm",
    "software",
    "compiler",
    "framework",
    "library",
    "database",
    "runtime",
    "protocol",
    "rust",
    "python",
    "java",
    "javascript",
    "typescript",
    "golang",
    "swift",
    "kotlin",
    "scala",
    "haskell",
    "lisp",
    "functional",
    "object-oriented",
    "systems",
    "embedded",
    "async",
    "concurrent",
    "machine learning",
    "deep learning",
    "neural",
    "blockchain",
    "kubernetes",
    "docker",
    "linux",
    "unix",
    "assembly",
    "llvm",
];

const PROGRAMMING_ASPECTS: &[&str] = &[
    "ownership model",
    "pattern matching",
    "type system",
    "memory safety",
    "concurrency model",
    "error handling",
    "trait objects",
    "generic programming",
    "functional programming",
    "async programming",
    "borrow checker",
    "package manager",
    "standard library",
    "systems programming",
    "compiler design",
    "zero-cost abstractions",
    "algebraic data types",
    "higher-order functions",
    "memory management",
    "embedded systems",
];

const GENERAL_ASPECTS: &[&str] = &[
    "history",
    "culture",
    "geography",
    "people",
    "economy",
    "architecture",
    "religion",
    "politics",
    "society",
    "notable",
    "overview",
    "characteristics",
    "development",
    "influence",
    "significance",
];

const STOPWORDS: &[&str] = &[
    "the", "this", "that", "these", "those", "a", "an", "and", "but", "or", "nor", "yet", "so",
    "for", "both", "of", "in", "on", "at", "to", "from", "with", "by", "as", "into", "over",
    "under", "about", "above", "below", "between", "through", "before", "after", "since", "until",
    "while", "within", "without", "it", "its", "they", "their", "them", "we", "our", "you", "your",
    "who", "which", "what", "when", "where", "how", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "do", "does", "did", "will", "would", "can", "could", "may",
    "might", "shall", "should", "must", "make", "made", "get", "gets", "got", "use", "uses",
    "used", "using", "not", "only", "just", "even", "also", "back", "well", "very", "first",
    "second", "third", "last", "next", "include", "includes", "however", "than", "then",
];

const FALSE_TOPIC_WORDS: &[&str] = &[
    "given",
    "axiom",
    "theorem",
    "lemma",
    "corollary",
    "proof",
    "definition",
    "statement",
    "conditional",
    "probability",
    "since",
    "however",
    "therefore",
    "thus",
    "hence",
    "whereas",
    "although",
    "despite",
    "section",
    "chapter",
    "figure",
    "note",
    "remark",
    "example",
    "case",
    "result",
    "summary",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "various",
    "several",
    "other",
    "another",
];

const GENERIC_NOUNS: &[&str] = &[
    "computing",
    "computer",
    "programming",
    "language",
    "languages",
    "software",
    "hardware",
    "system",
    "systems",
    "process",
    "processing",
    "data",
    "information",
    "technology",
    "implementation",
    "interface",
    "library",
    "framework",
    "runtime",
    "compiler",
    "interpreter",
    "modern",
    "advanced",
    "basic",
    "simple",
    "complex",
    "world",
    "country",
    "region",
    "area",
    "place",
    "city",
    "town",
    "people",
    "person",
    "group",
    "organization",
    "community",
    "society",
    "history",
    "culture",
    "time",
    "period",
    "century",
    "decade",
    "years",
    "research",
    "study",
    "field",
    "domain",
    "theory",
    "practice",
];

const PERSON_TITLES: &[&str] = &[
    "developer",
    "architect",
    "engineer",
    "researcher",
    "professor",
    "scientist",
    "creator",
    "inventor",
    "designer",
    "coder",
    "programmer",
    "author",
    "founder",
    "ceo",
    "cto",
    "phd",
    "student",
    "graduate",
    "undergraduate",
    "lecturer",
    "scholar",
    "musician",
    "singer",
    "songwriter",
    "artist",
    "painter",
    "composer",
    "writer",
    "journalist",
    "director",
    "producer",
    "actor",
    "actress",
    "politician",
    "president",
    "minister",
    "senator",
    "general",
];

const TECH_COMPOUNDS: &[&str] = &[
    "ownership model",
    "borrow checker",
    "type system",
    "type inference",
    "pattern matching",
    "algebraic data types",
    "higher-order functions",
    "memory safety",
    "memory management",
    "garbage collection",
    "concurrency model",
    "async runtime",
    "async programming",
    "trait objects",
    "zero-cost abstractions",
    "standard library",
    "systems programming",
    "compiler design",
    "error handling",
    "package manager",
];

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
        .unwrap_or_else(|| domain.split_whitespace().next().unwrap_or("knowledge"))
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
    if text.contains("spans many interconnected areas")
        || text.contains("spans many dimensions of knowledge")
        || text.contains("active area of research and practical application")
        || text.contains("continues to evolve with contributions")
        || text.contains("underpins key ideas in systems")
        || text.contains("widely studied across academia")
        || text.contains("foundational topic in modern software")
        || text.contains("spans many dimensions")
        || text.contains("is a foundational topic with many")
        || text.contains("spans many dimensions of knowledge")
        || text.contains("has shaped thought and practice")
        || text.contains("is widely explored across academic")
        || text.contains("continues to be studied and discussed")
    {
        return true;
    }
    if text.matches("...").count() >= 2 {
        return true;
    }
    if text.split_whitespace().count() < 20 {
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

    let first_sentences: Vec<&str> = clean.split(['.', '!', '?']).take(3).collect();

    for sentence in &first_sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        for (i, &word) in words.iter().enumerate() {
            if i == 0 || word.len() < 4 {
                continue;
            }
            if word.contains('-') || word.contains('_') {
                continue;
            }
            let core: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            if core.len() < 4 {
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
        if word.len() < 5 || word.contains('-') {
            continue;
        }
        let core: String = word.chars().filter(|c| c.is_alphabetic()).collect();
        if core.len() < 5 {
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
        .take(2)
        .collect();

    if content_words.len() >= 2 {
        let candidate = content_words.join(" ");
        if !candidate.eq_ignore_ascii_case(current_topic) {
            return candidate;
        }
    }

    aspect_topic(domain, turn + 1)
}
