// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Sentence Quality Filter
//!
//! Provides [`SentenceFilter`], a declarative rule-based classifier that rejects
//! low-quality sentences before they enter the [`KnowledgeIndex`] or the final
//! output assembly.
//!
//! ## Filter Rules (applied in order)
//!
//! | Rule | Condition | Action |
//! |---|---|---|
//! | Metadata | Contains known article-navigation markers | Reject |
//! | Header fragment | Matches `Title - Description` or `Title, Description` pattern | Reject |
//! | Disambiguation | Sentence begins with a noun followed by a Wikipedia definition fragment | Reject |
//! | Minimum length | Fewer than 8 words after tokenisation | Reject |
//! | Low information density | `unique_word_count / total_word_count < 0.55` | Reject |
//!
//! All rules use O(1) PHF lookups or simple string operations; the total cost
//! per sentence is O(|sentence|).
//!
//! [`KnowledgeIndex`]: crate::cognition::knowledge::KnowledgeIndex

use crate::constants::generation_filter::{
    ALL_CAPS_WORD_LIMIT, DEFAULT_EMPTY_WORD, LINK_WORD_LIMIT, MAX_CONSECUTIVE_TITLE,
    MIN_WORDS_FOR_QUESTION_PATTERN, MIN_WORDS_FOR_SENTENCE_LENGTH, NAVIGATION_LINK_LIMIT,
    NAVIGATION_WORD_LIMIT, QUESTION_PATTERN_WORD_LIMIT,
};
use phf::{Set, phf_set};
use std::collections::HashSet;

/// Compile-time set of words that mark Wikipedia navigation / metadata lines.
///
/// Any sentence that contains one of these tokens as a standalone word (case-
/// insensitive) is treated as a metadata artifact and rejected.
static METADATA_MARKERS: Set<&'static str> = phf_set! {
    "category",
    "categories",
    "references",
    "bibliography",
    "footnotes",
    "redirect",
    "disambiguation",
    "infobox",
    "navbox",
    "hatnote",
    "reflist",
    "cite",
};

/// Compile-time set of words that frequently appear in Wikipedia article-header
/// fragments that describe the article structure rather than its content.
static STRUCTURAL_MARKERS: Set<&'static str> = phf_set! {
    "outline",
    "overview",
    "topical",
    "baedeker",
    "admonisher",
    "abbreviature",
    "headpiece",
    "abbreviation",
};

/// Compile-time set of words that signal a Wikipedia disambiguation fragment.
///
/// These appear in sentences like _"Rust An iron oxide, a usually reddish-brown
/// oxide formed by the reaction of iron and oxygen in the presence of water."_
/// or _"Rust Belt An area of the United States..."_ that are attached to a
/// proper noun not relevant to the query.
static DISAMBIGUATION_SIGNALS: Set<&'static str> = phf_set! {
    "reddish",
    "brownish",
    "colloquially",
    "informally",
    "alternatively",
    "disambiguation",
    "homonym",
    "namesake",
    "eponym",
    "belt",
    "valley",
    "corridor",
    "triangle",
    "region",
    "district",
    "county",
    "province",
    "municipality",
    "borough",
    "township",
    "parish",
    "prefecture",
    "canton",
    "oxide",
    "corrosion",
    "fungus",
    "fungi",
    "pathogen",
    "parasitic",
    "troupe",
    "missile",
    "missiles",
    "insurance",
    "comedian",
    "comedy",
};

/// Compile-time set of first-words that make a sentence unsuitable as the
/// **opening sentence** of a paragraph.
///
/// These words presuppose context from a prior sentence:
/// - Pronouns: `"It"`, `"They"`, `"This"`, `"These"`, `"Those"`, etc.
/// - Additive / contrastive connectors: `"Also"`, `"Furthermore"`, `"By"`, etc.
///
/// A sentence whose first word (case-insensitive) is in this set is acceptable
/// as a *continuation* sentence but should not start a fresh paragraph.
static BAD_OPENERS: Set<&'static str> = phf_set! {
    "it", "its",
    "they", "them", "their", "theirs",
    "this", "these", "those",
    "he", "she", "we", "our",
    "also",
    "additionally",
    "furthermore",
    "moreover",
    "however",
    "nevertheless",
    "nonetheless",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "by",
    "beyond",
    "both",
    "another",
    "such",
    "other",
    "others",
};

/// Compile-time set of continuative adverbs that, when appearing as the
/// **second word** of a sentence, signal that the sentence presupposes a prior
/// sentence.
///
/// Example: `"Rust also provides..."`, first word is `"Rust"` (fine), but
/// second word `"also"` implies a list continuation.
static CONTINUATION_ADVERBS: Set<&'static str> = phf_set! {
    "also",
    "too",
    "still",
    "then",
    "again",
    "instead",
    "otherwise",
    "additionally",
    "similarly",
    "likewise",
};

/// Rule-based sentence quality classifier.
///
/// All methods are stateless and pure; `SentenceFilter` has no fields.
///
/// # Examples
///
/// ```rust
/// use lmm::generation::filter::SentenceFilter;
///
/// assert!(SentenceFilter::is_acceptable(
///     "Rust is a general-purpose programming language emphasizing memory safety."
/// ));
/// assert!(!SentenceFilter::is_acceptable(
///     "Rust (programming language) Category History of programming languages"
/// ));
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct SentenceFilter;

impl SentenceFilter {
    /// Returns `true` when `sentence` passes all quality gates and is suitable
    /// for indexing or inclusion in generated output.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A candidate sentence string.
    ///
    /// # Time Complexity
    ///
    /// O(|sentence|)
    pub fn is_acceptable(sentence: &str) -> bool {
        let first_char = sentence.chars().next().unwrap_or(' ');
        if first_char.is_ascii_digit() || first_char == ',' || first_char == ';' {
            return false;
        }
        Self::has_minimum_length(sentence)
            && !Self::is_metadata_line(sentence)
            && !Self::is_header_fragment(sentence)
            && !Self::is_disambiguation_fragment(sentence)
            && !Self::is_title_prefix_fragment(sentence)
            && !Self::is_bare_question_title(sentence)
            && !Self::is_low_density(sentence)
    }

    /// Returns `true` when this sentence can serve as the **opening sentence**
    /// of a paragraph without presupposing prior context.
    ///
    /// Two conditions are checked:
    ///
    /// 1. **First word** must not be a pronoun or additive connector that
    ///    references prior context (`"It"`, `"They"`, `"Also"`, `"By"`, ...).
    /// 2. **Second word** must not be a continuative adverb such as `"also"`,
    ///    `"too"`, or `"still"`, patterns like `"Rust also provides..."` or
    ///    `"Python still supports..."` imply a prior sentence exists.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::filter::SentenceFilter;
    ///
    /// assert!(SentenceFilter::is_good_opener(
    ///     "Rust is a general-purpose programming language emphasizing memory safety."
    /// ));
    /// assert!(!SentenceFilter::is_good_opener(
    ///     "It was influenced by ideas from functional programming and type theory."
    /// ));
    /// assert!(!SentenceFilter::is_good_opener(
    ///     "Rust also provides traits and methods as alternatives to inheritance."
    /// ));
    /// ```
    ///
    /// # Time Complexity
    ///
    /// O(1): inspects only the first two words.
    pub fn is_good_opener(sentence: &str) -> bool {
        let mut words = sentence.split_whitespace();
        let first = words
            .next()
            .unwrap_or(DEFAULT_EMPTY_WORD)
            .trim_matches(|c: char| !c.is_ascii_alphabetic())
            .to_ascii_lowercase();
        if BAD_OPENERS.contains(first.as_str()) {
            return false;
        }
        let second = words
            .next()
            .unwrap_or(DEFAULT_EMPTY_WORD)
            .trim_matches(|c: char| !c.is_ascii_alphabetic())
            .to_ascii_lowercase();
        !CONTINUATION_ADVERBS.contains(second.as_str())
    }

    /// Returns `true` when the sentence has at least 8 words.
    ///
    /// # Time Complexity
    ///
    /// O(|sentence|)
    pub fn has_minimum_length(sentence: &str) -> bool {
        sentence.split_whitespace().count() >= MIN_WORDS_FOR_SENTENCE_LENGTH
    }

    /// Returns `true` when the sentence contains a known metadata marker token.
    ///
    /// Tokens are matched case-insensitively against [`METADATA_MARKERS`] and
    /// [`STRUCTURAL_MARKERS`].  The check also handles the common Wikipedia
    /// `Category:` and `Category :` prefix forms, where the colon is immediately
    /// attached to the word without whitespace separation.
    ///
    /// # Time Complexity
    ///
    /// O(|sentence|): one PHF lookup per word.
    pub fn is_metadata_line(sentence: &str) -> bool {
        for word in sentence.split_whitespace() {
            let lower = word
                .trim_matches(|c: char| !c.is_ascii_alphabetic())
                .to_ascii_lowercase();
            if METADATA_MARKERS.contains(lower.as_str())
                || STRUCTURAL_MARKERS.contains(lower.as_str())
            {
                return true;
            }
            if let Some(prefix) = word.split(':').next() {
                let prefix_lower = prefix
                    .trim_matches(|c: char| !c.is_ascii_alphabetic())
                    .to_ascii_lowercase();
                if METADATA_MARKERS.contains(prefix_lower.as_str())
                    || STRUCTURAL_MARKERS.contains(prefix_lower.as_str())
                {
                    return true;
                }
            }
        }
        false
    }

    /// Returns `true` when the sentence matches the Wikipedia article-header
    /// pattern: a proper noun followed by ` - ` or ` — ` and additional text
    /// **within the first 80 characters**, which is where page-title separators
    /// appear.  Legitimate body sentences that happen to contain an em-dash or
    /// hyphen in the middle of a clause are not affected.
    ///
    /// Also detects the DuckDuckGo related-topic fragment pattern:
    /// `"Name (qualifier) Description"`, a proper noun, a parenthesised
    /// qualifier, and an un-verbified description in the same sentence,
    /// e.g. `"Python (missile) A family of air-to-air missiles built by..."`.
    ///
    /// Example: `"Outline of the Rust programming language - The following outline..."`
    ///
    /// # Time Complexity
    ///
    /// O(min(|sentence|, 80))
    pub fn is_header_fragment(sentence: &str) -> bool {
        let preview = if sentence.len() >= 100 {
            sentence
                .char_indices()
                .nth(100)
                .map(|(i, _)| &sentence[..i])
                .unwrap_or(sentence)
        } else {
            sentence
        };
        if preview.contains(" - ") || preview.contains(" — ") {
            return true;
        }
        if let Some(pos) = sentence.find('\u{2014}')
            && pos < 30
        {
            return true;
        }
        let bytes = sentence.as_bytes();
        if bytes.first().is_some_and(|b| b.is_ascii_uppercase())
            && let Some(open) = sentence.find(" (")
            && open < 40
            && sentence[open..].contains(") ")
        {
            return true;
        }
        false
    }

    /// Returns `true` when the sentence contains a disambiguation signal word,
    /// indicating it describes a homonym of the queried topic rather than the
    /// topic itself.
    ///
    /// Example: `"Rust An iron oxide, a usually reddish-brown oxide..."`
    ///
    /// Hyphenated tokens (e.g. `"reddish-brown"`) are split on the hyphen and
    /// each component is checked individually against [`DISAMBIGUATION_SIGNALS`].
    ///
    /// # Time Complexity
    ///
    /// O(|sentence|): one PHF lookup per word.
    pub fn is_disambiguation_fragment(sentence: &str) -> bool {
        for word in sentence.split_whitespace() {
            for part in word.split('-') {
                let lower = part
                    .trim_matches(|c: char| !c.is_ascii_alphabetic())
                    .to_ascii_lowercase();
                if DISAMBIGUATION_SIGNALS.contains(lower.as_str()) {
                    return true;
                }
            }
        }
        false
    }

    /// Returns `true` when the sentence matches the Wikipedia abstract-title
    /// prefix pattern: `"<Topic Title> <Abstract without verb link>"`.
    ///
    /// DuckDuckGo Instant Answer sometimes returns `abstract_text` values that
    /// begin with the article title followed directly by its definition body,
    /// without a linking verb.  For example:
    ///
    /// > `"Artificial intelligence The capability of computational systems..."`
    ///
    /// These are detected by checking whether the first capitalised word(s) are
    /// immediately followed by another capitalised word with no linking verb
    /// (`is`, `are`, `was`, `were`, `can`, `has`, `have`) in the first 6 words.
    ///
    /// # Time Complexity
    ///
    /// O(1): inspects only the first 6 words.
    pub fn is_title_prefix_fragment(sentence: &str) -> bool {
        static LINK_WORDS: Set<&'static str> = phf_set! {
            "is", "are", "was", "were", "can", "could", "has", "have",
            "had", "will", "would", "may", "might", "shall", "should",
            "does", "do", "did", "a", "an", "the", "of", "in", "to",
        };
        let words: Vec<&str> = sentence
            .split_whitespace()
            .take(QUESTION_PATTERN_WORD_LIMIT)
            .collect();
        if words.len() < MIN_WORDS_FOR_QUESTION_PATTERN {
            return false;
        }
        let clean = |w: &str| {
            w.trim_matches(|c: char| !c.is_ascii_alphabetic())
                .to_ascii_lowercase()
        };

        let w1 = clean(words[1]);
        let w2 = clean(words[2]);
        if !LINK_WORDS.contains(w1.as_str()) && (w2 == "the" || w2 == "a" || w2 == "an") {
            return true;
        }

        let all_caps = words
            .iter()
            .take(ALL_CAPS_WORD_LIMIT)
            .all(|w| w.chars().next().is_some_and(|c| c.is_uppercase()));
        if all_caps {
            let has_link = words
                .iter()
                .take(LINK_WORD_LIMIT)
                .any(|w| LINK_WORDS.contains(clean(w).as_str()));
            if !has_link {
                return true;
            }
        }

        let paren_count = sentence.match_indices(" (").count();
        if paren_count >= 2 {
            return true;
        }

        let mut consecutive_title = 0usize;
        let mut max_consec = 0usize;
        for w in sentence.split_whitespace().take(NAVIGATION_WORD_LIMIT) {
            if w.chars().next().is_some_and(|c| c.is_uppercase()) {
                consecutive_title += 1;
                max_consec = max_consec.max(consecutive_title);
            } else {
                consecutive_title = 0;
            }
        }
        if max_consec >= MAX_CONSECUTIVE_TITLE {
            let has_link_early = sentence
                .split_whitespace()
                .take(NAVIGATION_LINK_LIMIT)
                .any(|w| {
                    LINK_WORDS.contains(
                        w.trim_matches(|c: char| !c.is_ascii_alphabetic())
                            .to_ascii_lowercase()
                            .as_str(),
                    )
                });
            if !has_link_early {
                return true;
            }
        }

        let paren_count = sentence.match_indices(" (").count();
        if paren_count >= 2 {
            return true;
        }

        let mut consecutive_title = 0usize;
        let mut max_consec = 0usize;
        for w in sentence.split_whitespace().take(NAVIGATION_WORD_LIMIT) {
            if w.chars().next().is_some_and(|c| c.is_uppercase()) {
                consecutive_title += 1;
                max_consec = max_consec.max(consecutive_title);
            } else {
                consecutive_title = 0;
            }
        }
        if max_consec >= MAX_CONSECUTIVE_TITLE {
            let has_link_early = sentence
                .split_whitespace()
                .take(NAVIGATION_LINK_LIMIT)
                .any(|w| {
                    LINK_WORDS.contains(
                        w.trim_matches(|c: char| !c.is_ascii_alphabetic())
                            .to_ascii_lowercase()
                            .as_str(),
                    )
                });
            if !has_link_early {
                return true;
            }
        }

        false
    }

    /// Returns `true` when the sentence is a bare search-query title that starts
    /// with a question word (`What`, `Why`, `How`, `When`, `Where`, `Which`, `Who`)
    /// and does not end with a question mark.  Such strings are DuckDuckGo
    /// related-topic link titles, not prose sentences.
    ///
    /// Example: `"What does Rust have instead of a garbage collector"`, no `?`,
    /// starts with `"What"`, qualifies as a bare query fragment.
    ///
    /// # Time Complexity
    ///
    /// O(1): inspects only the first word and last character.
    pub fn is_bare_question_title(sentence: &str) -> bool {
        static QUESTION_STARTERS: &[&str] = &[
            "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
        ];
        let first = sentence
            .split_whitespace()
            .next()
            .unwrap_or(DEFAULT_EMPTY_WORD)
            .trim_matches(|c: char| !c.is_ascii_alphabetic())
            .to_ascii_lowercase();
        let ends_with_question = sentence.trim_end().ends_with('?');
        QUESTION_STARTERS.contains(&first.as_str()) && !ends_with_question
    }

    /// Returns `true` when fewer than 50% of the sentence's words are unique,
    /// indicating a list-like, highly repetitive, or navigational structure.
    ///
    /// # Time Complexity
    ///
    /// O(|sentence|)
    pub fn is_low_density(sentence: &str) -> bool {
        let words: Vec<String> = sentence
            .split_whitespace()
            .map(|w| w.to_ascii_lowercase())
            .collect();
        let total = words.len();
        if total == 0 {
            return true;
        }
        let unique: HashSet<&String> = words.iter().collect();
        let density = unique.len() as f64 / total as f64;
        density < 0.50
    }
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
