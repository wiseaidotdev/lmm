// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Poem Responder
//!
//! Composes original poems from a KB corpus gathered about a subject.
//!
//! [`PoemResponder::compose`] splits the corpus into sentences, selects lines
//! that mention the subject, trims each line to a poetic length, groups them
//! into stanzas, and assembles the full poem.  When the corpus is too sparse
//! the responder falls back to built-in template lines containing the subject.

use crate::constants::poem_responder::{
    POEM_FALLBACK_LINES, POEM_JUNK_PREFIX_RETRIEVED, POEM_JUNK_PREFIX_SEARCH,
    POEM_JUNK_PREFIX_STARS, POEM_LINES_PER_STANZA, POEM_MAX_LINE_WORDS, POEM_MAX_STANZAS,
    POEM_MIN_ALPHA_RATIO_DENOMINATOR, POEM_MIN_SENTENCE_WORDS, POEM_STANZA_SEPARATOR,
};

/// Composes original poems from KB corpus text gathered about a subject.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::poem_responder::PoemResponder;
///
/// let poem = PoemResponder::compose("life", "Life is a journey. Every step matters.", 42);
/// assert!(!poem.is_empty());
/// ```
pub struct PoemResponder;

impl PoemResponder {
    /// Composes a poem about `subject` from `corpus` text using `seed` for variation.
    ///
    /// 1. Splits `corpus` into sentences.
    /// 2. Keeps sentences that mention `subject` or are long enough to serve as a line.
    /// 3. Trims each sentence to at most [`POEM_MAX_LINE_WORDS`] words.
    /// 4. Groups lines into stanzas of [`POEM_LINES_PER_STANZA`] lines.
    /// 5. Falls back to [`POEM_FALLBACK_LINES`] when too few corpus lines are found.
    ///
    /// # Arguments
    ///
    /// * `subject` - The domain topic the poem should be about.
    /// * `corpus`  - KB text gathered from web/skill retrieval.
    /// * `seed`    - Per-run seed used to offset which fallback lines are selected.
    ///
    /// # Time Complexity
    ///
    /// O(|corpus|)
    pub fn compose(subject: &str, corpus: &str, seed: u64) -> String {
        let subject_lower = subject.to_ascii_lowercase();
        let raw_lines = Self::extract_lines(corpus, &subject_lower);
        let lines = if raw_lines.len() >= POEM_LINES_PER_STANZA {
            raw_lines
        } else {
            Self::fallback_lines(subject, seed)
        };
        Self::assemble_stanzas(&lines)
    }

    /// Splits `corpus` into candidate poem lines.
    ///
    /// A candidate sentence is kept when it mentions `subject_lower` or has
    /// at least [`POEM_MIN_SENTENCE_WORDS`] words (generic descriptive content).
    /// Each line is then word-truncated to [`POEM_MAX_LINE_WORDS`].
    ///
    /// # Time Complexity
    ///
    /// O(|corpus|)
    fn extract_lines(corpus: &str, subject_lower: &str) -> Vec<String> {
        corpus
            .split(['.', '!', '?', '\n'])
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return None;
                }
                if !Self::is_clean_prose_line(trimmed) {
                    return None;
                }
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                let lower = trimmed.to_ascii_lowercase();
                let relevant =
                    lower.contains(subject_lower) || words.len() >= POEM_MIN_SENTENCE_WORDS;
                if !relevant {
                    return None;
                }
                let truncated: String = words
                    .into_iter()
                    .take(POEM_MAX_LINE_WORDS)
                    .collect::<Vec<_>>()
                    .join(" ");
                Some(truncated)
            })
            .take(POEM_MAX_STANZAS * POEM_LINES_PER_STANZA)
            .collect()
    }

    /// Returns `true` when `s` looks like clean prose rather than HTML or navigation junk.
    ///
    /// # Time Complexity
    ///
    /// O(|s|)
    fn is_clean_prose_line(s: &str) -> bool {
        let lower = s.to_ascii_lowercase();
        !s.contains('<')
            && !s.contains('>')
            && !s.contains("http")
            && !s.contains("://")
            && !s.contains('%')
            && !lower.starts_with('#')
            && !lower.starts_with(POEM_JUNK_PREFIX_RETRIEVED)
            && !lower.starts_with(POEM_JUNK_PREFIX_SEARCH)
            && !lower.starts_with(POEM_JUNK_PREFIX_STARS)
            && s.chars().filter(|c| c.is_ascii_alphabetic()).count()
                > s.len() / POEM_MIN_ALPHA_RATIO_DENOMINATOR
    }

    /// Generates fallback poem lines by filling `{subject}` placeholders.
    ///
    /// Uses `seed` to rotate through available templates so repeated calls
    /// produce different stanza selections.
    ///
    /// # Time Complexity
    ///
    /// O(|POEM_FALLBACK_LINES|)
    fn fallback_lines(subject: &str, seed: u64) -> Vec<String> {
        let offset = (seed as usize) % POEM_FALLBACK_LINES.len();
        POEM_FALLBACK_LINES
            .iter()
            .cycle()
            .skip(offset)
            .take(POEM_MAX_STANZAS * POEM_LINES_PER_STANZA)
            .map(|tmpl| tmpl.replace("{subject}", subject))
            .collect()
    }

    /// Groups `lines` into stanzas separated by [`POEM_STANZA_SEPARATOR`].
    ///
    /// # Time Complexity
    ///
    /// O(|lines|)
    fn assemble_stanzas(lines: &[String]) -> String {
        lines
            .chunks(POEM_LINES_PER_STANZA)
            .take(POEM_MAX_STANZAS)
            .map(|stanza| stanza.join("\n"))
            .collect::<Vec<_>>()
            .join(POEM_STANZA_SEPARATOR)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
