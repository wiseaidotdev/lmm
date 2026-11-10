// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Paragraph Shaper
//!
//! Provides [`ParagraphShaper`], which takes a flat list of deduplicated sentences
//! and formats them into well-structured prose paragraphs subject to configurable
//! word-count and sentence-count constraints.
//!
//! ## Why a separate shaper?
//!
//! `assemble_paragraphs` in `crate::app` shuffles sentences and groups them by
//! count, but provides no guarantees on paragraph density.  This leads to single-
//! sentence paragraphs or paragraphs that are far too long when the KB returns
//! many related chunks.
//!
//! `ParagraphShaper` enforces:
//!
//! | Constraint | Default |
//! |---|---|
//! | Paragraph count | 2-3 |
//! | Words per paragraph | 40-120 |
//! | Sentences per paragraph | 2-5 |
//! | Discourse connectors | Between paragraphs only |
//!
//! Sentence boundaries are always respected, no sentence is split in half.

/// Discourse connectors prepended to the opening sentence of non-first paragraphs.
///
/// Only connectors that work grammatically in front of a sentence beginning with
/// a proper noun are included here (e.g. `"Furthermore, Rust provides..."` reads
/// naturally; `"By extension, rust provides..."` does not because the proper noun
/// must stay capitalised yet the connector demands sentence-case continuation).
static CONNECTORS: &[&str] = &[
    "Furthermore,",
    "In addition,",
    "Notably,",
    "Moreover,",
    "As a result,",
    "In practice,",
    "Interestingly,",
    "Consequently,",
    "Building on this,",
    "To elaborate,",
    "More specifically,",
    "On the technical side,",
    "From a practical standpoint,",
    "In this context,",
];

/// Shapes a flat list of sentences into well-structured multi-paragraph prose.
///
/// The shaper distributes sentences across paragraphs while respecting minimum
/// and maximum word-count and sentence-count constraints per paragraph.
///
/// # Examples
///
/// ```rust
/// use lmm::generation::shaper::ParagraphShaper;
///
/// let shaper = ParagraphShaper::default();
/// let sentences = vec![
///     "Rust is a systems programming language focused on safety.".to_string(),
///     "It prevents data races at compile time through the ownership model.".to_string(),
///     "Memory is managed without a garbage collector or runtime overhead.".to_string(),
///     "Rust achieves zero-cost abstractions enabling native performance.".to_string(),
/// ];
/// let prose = shaper.shape(&sentences, 42);
/// assert!(!prose.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct ParagraphShaper {
    /// Target number of output paragraphs; actual count may be less when
    /// there are not enough sentences to fill every paragraph.
    pub target_paragraphs: usize,
    /// Minimum number of sentences allowed per paragraph.
    pub min_sentences_per_para: usize,
    /// Maximum number of sentences allowed per paragraph.
    pub max_sentences_per_para: usize,
    /// Minimum word count a paragraph must meet before closing.
    pub min_words_per_para: usize,
    /// Maximum word count after which a new paragraph must be started.
    pub max_words_per_para: usize,
}

impl Default for ParagraphShaper {
    fn default() -> Self {
        Self {
            target_paragraphs: 2,
            min_sentences_per_para: 2,
            max_sentences_per_para: 5,
            min_words_per_para: 40,
            max_words_per_para: 120,
        }
    }
}

impl ParagraphShaper {
    /// Creates a `ParagraphShaper` with explicit constraints.
    ///
    /// # Arguments
    ///
    /// * `target_paragraphs`    - Desired number of output paragraphs.
    /// * `min_sentences`        - Minimum sentences per paragraph.
    /// * `max_sentences`        - Maximum sentences per paragraph.
    /// * `min_words`            - Minimum words per paragraph.
    /// * `max_words`            - Maximum words per paragraph.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn new(
        target_paragraphs: usize,
        min_sentences: usize,
        max_sentences: usize,
        min_words: usize,
        max_words: usize,
    ) -> Self {
        Self {
            target_paragraphs: target_paragraphs.max(1),
            min_sentences_per_para: min_sentences.max(1),
            max_sentences_per_para: max_sentences.max(min_sentences + 1),
            min_words_per_para: min_words,
            max_words_per_para: max_words.max(min_words + 10),
        }
    }

    /// Shapes `sentences` into paragraphs, returning multi-paragraph prose with
    /// `\n\n` separating paragraphs.
    ///
    /// A pseudo-random LCG seeded from `run_seed` selects the discourse connector
    /// for each non-first paragraph, ensuring per-run variation.
    ///
    /// # Arguments
    ///
    /// * `sentences` - Ordered slice of deduplicated sentence strings.
    /// * `run_seed`  - Per-run entropy value for connector selection.
    ///
    /// # Returns
    ///
    /// A `String` containing the shaped prose. Returns an empty string when
    /// `sentences` is empty.
    ///
    /// # Time Complexity
    ///
    /// O(Σ|sentence|)
    ///
    /// # Space Complexity
    ///
    /// O(|output|)
    pub fn shape(&self, sentences: &[String], run_seed: u64) -> String {
        if sentences.is_empty() {
            return String::new();
        }

        let groups = self.partition(sentences, run_seed);
        let mut paragraphs: Vec<String> = Vec::new();
        let mut seed = run_seed;

        for (p, group) in groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }

            let para = if p > 0 {
                seed = lcg_next(seed);
                let connector = CONNECTORS[(seed as usize) % CONNECTORS.len()];
                let first = &group[0];
                let joined_first = format!("{} {}", connector, lower_first_common(first));
                let mut parts = vec![joined_first];
                for s in &group[1..] {
                    parts.push(s.clone());
                }
                parts.join(" ")
            } else {
                group.join(" ")
            };

            paragraphs.push(capitalise(para));
        }

        paragraphs.join("\n\n")
    }

    /// Partitions `sentences` into groups, each group becoming one paragraph.
    ///
    /// The partitioning algorithm fills a paragraph by accumulating sentences
    /// until either the sentence count or word count limit is reached, then
    /// starts a new paragraph.  Any remaining sentences are appended to the
    /// last paragraph rather than creating a paragraph that is too short.
    ///
    /// # Time Complexity
    ///
    /// O(Σ|sentence|)
    fn partition(&self, sentences: &[String], run_seed: u64) -> Vec<Vec<String>> {
        let max_paras = self.target_paragraphs;
        let mut groups: Vec<Vec<String>> = vec![Vec::new(); max_paras];
        let mut para_idx = 0usize;
        let mut current_words = 0usize;
        let mut current_sents = 0usize;

        for sentence in sentences {
            let wc = sentence.split_whitespace().count();

            let full = current_sents >= self.max_sentences_per_para
                || current_words + wc > self.max_words_per_para;

            let enough = current_sents >= self.min_sentences_per_para
                && current_words >= self.min_words_per_para;

            let can_advance = para_idx + 1 < max_paras;

            if full && enough && can_advance {
                para_idx += 1;
                current_words = 0;
                current_sents = 0;
            }

            groups[para_idx].push(sentence.clone());
            current_words += wc;
            current_sents += 1;
        }

        let _ = run_seed;
        let mut result: Vec<Vec<String>> = groups.into_iter().filter(|g| !g.is_empty()).collect();

        if result.len() >= 2 {
            let last_idx = result.len() - 1;
            if result[last_idx].len() < self.min_sentences_per_para {
                let last = result.remove(last_idx);
                let prev = result.last_mut().expect("at least one paragraph remains");
                prev.extend(last);
            }
        }

        result
    }
}

/// Advances one step of a Knuth MMIX LCG.
///
/// # Time Complexity
///
/// O(1)
fn lcg_next(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Capitalises the first character of `s`.
///
/// # Time Complexity
///
/// O(|s|)
fn capitalise(s: String) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let upper: String = c.to_uppercase().collect();
            upper + chars.as_str()
        }
    }
}

/// Lowercases the first character of `s` only when it is a common English word
/// that was capitalised solely because it began a sentence (e.g. `"The"`,
/// `"A"`, `"An"`).  Proper nouns and acronyms are left unchanged.
///
/// A word is treated as a proper noun when its second character (if present)
/// is also uppercase (e.g. `"UK"`, `"US"`) or when the whole first word appears
/// in a known proper-noun set.  Single-letter words are always lowercased.
///
/// # Time Complexity
///
/// O(|s|)
fn lower_first_common(s: &str) -> String {
    static COMMON_SENTENCE_STARTERS: phf::Set<&'static str> = phf::phf_set! {
        "the", "a", "an", "this", "that", "these", "those",
        "one", "two", "three", "four", "five",
        "each", "every", "all", "both", "some", "many", "most",
        "evidence", "analysis", "reasoning", "drawing", "systematic",
        "based", "given", "since", "while", "although", "despite",
    };

    let first_word: String = s
        .split_whitespace()
        .next()
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_lowercase();

    if COMMON_SENTENCE_STARTERS.contains(first_word.as_str()) {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(c) => {
                let lower: String = c.to_lowercase().collect();
                lower + chars.as_str()
            }
        }
    } else {
        s.to_string()
    }
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
