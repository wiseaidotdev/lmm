// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Global Sentence Deduplicator
//!
//! Provides [`GlobalDeduplicator`], which removes near-duplicate sentences from
//! a collection using **Jaccard similarity** on content-word token sets.
//!
//! Exact prefix matching (the existing `deduplicate_sentences` in
//! `crate::text`) catches only verbatim repeats.  However the TRAM pipeline
//! assembles text from multiple sourcesm, KB extractive answer, expert inference
//! templates, ThinkLoop cold memory, that can produce semantically equivalent
//! sentences with minor wording differences:
//!
//! ```text
//! "Evidence from the knowledge base confirms that rust is multi-paradigm."
//! "Analysis of the available knowledge shows that rust is multi-paradigm."
//! ```
//!
//! Both sentences share the same factual claim.  Jaccard similarity on their
//! content-word token sets exceeds 0.70, so the deduplicator retains the first
//! and discards the second.
//!
//! ## Algorithm (O(n² × |sentence|))
//!
//! 1. Tokenise each sentence into a `HashSet<String>` of lowercase words ≥ 3 chars,
//!    excluding stop words.
//! 2. For each candidate sentence, compute Jaccard similarity against every
//!    previously accepted sentence.
//! 3. Accept the candidate only when no accepted sentence has similarity ≥ `threshold`.

use std::collections::HashSet;

/// Compile-time set of stop words excluded from Jaccard token sets.
static STOP_WORDS_DEDUP: phf::Set<&'static str> = phf::phf_set! {
    "the", "and", "for", "are", "was", "that", "this", "with", "from", "have",
    "not", "but", "had", "has", "its", "were", "they", "will", "been", "their",
    "all", "one", "can", "her", "his", "him", "she", "who", "which", "what",
    "into", "then", "than", "when", "also", "more", "some", "out", "about",
    "said", "would", "could", "should", "each", "other", "there", "these",
    "those", "such", "any", "our", "you", "your", "very", "just", "now", "may",
    "upon", "over", "under", "both", "few", "because", "while"
};

/// Removes near-duplicate sentences from a list using Jaccard similarity.
///
/// Two sentences are considered duplicates when their content-word token-set
/// Jaccard similarity is at or above `similarity_threshold`.  The first
/// occurrence of each near-duplicate group is retained; subsequent occurrences
/// are discarded.
///
/// # Examples
///
/// ```rust
/// use lmm::generation::dedup::GlobalDeduplicator;
///
/// let dedup = GlobalDeduplicator::new(0.70);
/// let sentences = vec![
///     "Rust is a general-purpose programming language emphasizing memory safety.".to_string(),
///     "Rust is a general-purpose programming language emphasizing memory safety.".to_string(),
///     "Python uses dynamic typing and garbage collection.".to_string(),
/// ];
/// let result = dedup.deduplicate(sentences);
/// assert_eq!(result.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct GlobalDeduplicator {
    /// Jaccard similarity threshold above which two sentences are considered duplicates.
    pub similarity_threshold: f64,
}

impl GlobalDeduplicator {
    /// Creates a new `GlobalDeduplicator`.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Jaccard threshold ∈ [0.0, 1.0]; values ≥ this threshold
    ///   are treated as duplicates.  Recommended range: 0.65-0.75.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn new(threshold: f64) -> Self {
        Self {
            similarity_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Filters `sentences`, retaining only sentences that are not Jaccard-similar
    /// to any previously retained sentence.
    ///
    /// # Arguments
    ///
    /// * `sentences` - Owned list of sentence strings.
    ///
    /// # Returns
    ///
    /// A new `Vec<String>` with near-duplicates removed.
    ///
    /// # Time Complexity
    ///
    /// O(n² × |sentence|) where n is the number of sentences.
    ///
    /// # Space Complexity
    ///
    /// O(n × |tokens_per_sentence|)
    pub fn deduplicate(&self, sentences: Vec<String>) -> Vec<String> {
        let mut accepted: Vec<String> = Vec::new();
        let mut accepted_tokens: Vec<HashSet<String>> = Vec::new();

        for sentence in sentences {
            let tokens = tokenise_for_jaccard(&sentence);
            let is_duplicate = accepted_tokens
                .iter()
                .any(|existing| jaccard(&tokens, existing) >= self.similarity_threshold);
            if !is_duplicate {
                accepted_tokens.push(tokens);
                accepted.push(sentence);
            }
        }

        accepted
    }

    /// Splits `text` on sentence boundaries, deduplicates the resulting
    /// sentences, and rejoins them preserving the original paragraph breaks
    /// (`\n\n`).
    ///
    /// # Arguments
    ///
    /// * `text` - Multi-paragraph prose string.
    ///
    /// # Time Complexity
    ///
    /// O(n² × |sentence|)
    pub fn deduplicate_text(&self, text: &str) -> String {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let all_sentences: Vec<String> = paragraphs
            .iter()
            .flat_map(|para| split_to_sentences(para))
            .collect();

        let deduplicated = self.deduplicate(all_sentences);
        reassemble_paragraphs(deduplicated, paragraphs.len())
    }
}

/// Splits text into sentences on `.`, `!`, `?` boundaries, keeping the
/// punctuation attached to the preceding sentence.
///
/// # Time Complexity
///
/// O(|text|)
fn split_to_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let _len = chars.len();

    for (i, &ch) in chars.iter().enumerate() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let next_is_lower = chars
                .get(i + 1)
                .map(|c| c.is_ascii_lowercase())
                .unwrap_or(false);
            let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_is_digit = chars
                .get(i + 1)
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false);
            if ch == '.' && (next_is_lower || (prev_is_digit && next_is_digit)) {
                continue;
            }
            let s = current.trim().to_string();
            if s.split_whitespace().count() >= 5 {
                sentences.push(s);
            }
            current.clear();
        }
    }

    let tail = current.trim().to_string();
    if tail.split_whitespace().count() >= 5 {
        sentences.push(tail);
    }

    sentences
}

/// Reassembles a flat list of sentences back into multi-paragraph prose.
///
/// Sentences are distributed evenly across `target_para_count` paragraphs,
/// separated by `\n\n`.
///
/// # Time Complexity
///
/// O(n × |sentence|)
fn reassemble_paragraphs(sentences: Vec<String>, target_para_count: usize) -> String {
    if sentences.is_empty() {
        return String::new();
    }
    let para_count = target_para_count.clamp(1, sentences.len());
    let per_para = (sentences.len() / para_count).max(1);
    let mut paragraphs: Vec<String> = Vec::new();
    let mut offset = 0usize;

    for p in 0..para_count {
        let take = if p == para_count - 1 {
            sentences.len().saturating_sub(offset)
        } else {
            per_para
        };
        if take == 0 || offset >= sentences.len() {
            break;
        }
        let end = (offset + take).min(sentences.len());
        paragraphs.push(sentences[offset..end].join(" "));
        offset = end;
    }

    paragraphs.join("\n\n")
}

/// Returns the Jaccard similarity between two token sets.
///
/// `J(A, B) = |A ∩ B| / |A ∪ B|`
///
/// Returns 1.0 when both sets are empty.
///
/// # Time Complexity
///
/// O(min(|a|, |b|))
pub fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.iter().filter(|t| b.contains(*t)).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Tokenises `sentence` into lowercase content words (≥ 3 chars, non-stop-word)
/// for Jaccard similarity computation.
///
/// # Time Complexity
///
/// O(|sentence|)
pub fn tokenise_for_jaccard(sentence: &str) -> HashSet<String> {
    sentence
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter_map(|w| {
            let w = w.to_ascii_lowercase();
            if w.len() >= 3 && !STOP_WORDS_DEDUP.contains(w.as_str()) {
                Some(w)
            } else {
                None
            }
        })
        .collect()
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
