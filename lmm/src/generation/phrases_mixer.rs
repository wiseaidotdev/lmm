// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Phrases Mixer
//!
//! Blends academic phrase banks from [`crate::constants::phrases`] into
//! generated response text so that TRAM output reads as fresh, articulate
//! prose rather than a verbatim excerpt from the knowledge corpus.
//!
//! ## Design
//!
//! * **Zero-allocation selection**: phrase banks are `'static` slices;
//!   selection is done with modular arithmetic (`seed % len`).
//! * **Singleton access**: [`PhrasesMixer::global`] returns a `&'static`
//!   reference via [`std::sync::OnceLock`], so initialisation cost is paid
//!   exactly once.
//! * **Idempotent enrichment**: paragraphs that already start with an
//!   academic opener are left untouched.

use crate::constants::phrases::{
    ACADEMIC_HEDGES, ACADEMIC_OPENERS, CONCLUSION_STARTERS, DISCOURSE_MARKERS, TRANSITION_PHRASES,
};
use std::sync::OnceLock;

/// A bank-aware phrase selector and paragraph enricher.
///
/// Obtain the process-wide singleton via [`PhrasesMixer::global`].
///
/// # Examples
///
/// ```rust
/// use lmm::generation::phrases_mixer::PhrasesMixer;
///
/// let m = PhrasesMixer::global();
/// let opener = m.opener(42);
/// assert!(!opener.is_empty());
/// ```
pub struct PhrasesMixer {
    openers: &'static [&'static str],
    markers: &'static [&'static str],
    transitions: &'static [&'static str],
    conclusions: &'static [&'static str],
    hedges: &'static [&'static str],
}

static GLOBAL: OnceLock<PhrasesMixer> = OnceLock::new();

impl PhrasesMixer {
    /// Returns the process-wide [`PhrasesMixer`] singleton.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::phrases_mixer::PhrasesMixer;
    ///
    /// let _m = PhrasesMixer::global();
    /// ```
    pub fn global() -> &'static Self {
        GLOBAL.get_or_init(|| Self {
            openers: ACADEMIC_OPENERS,
            markers: DISCOURSE_MARKERS,
            transitions: TRANSITION_PHRASES,
            conclusions: CONCLUSION_STARTERS,
            hedges: ACADEMIC_HEDGES,
        })
    }

    /// Returns an academic opener phrase selected deterministically from `seed`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::phrases_mixer::PhrasesMixer;
    ///
    /// let phrase = PhrasesMixer::global().opener(0);
    /// assert!(!phrase.is_empty());
    /// ```
    pub fn opener(&self, seed: u64) -> &'static str {
        self.openers[seed as usize % self.openers.len()]
    }

    /// Returns a discourse marker phrase selected deterministically from `seed`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::phrases_mixer::PhrasesMixer;
    ///
    /// let phrase = PhrasesMixer::global().marker(7);
    /// assert!(!phrase.is_empty());
    /// ```
    pub fn marker(&self, seed: u64) -> &'static str {
        self.markers[seed as usize % self.markers.len()]
    }

    /// Returns a transition phrase selected deterministically from `seed`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::phrases_mixer::PhrasesMixer;
    ///
    /// let phrase = PhrasesMixer::global().transition(3);
    /// assert!(!phrase.is_empty());
    /// ```
    pub fn transition(&self, seed: u64) -> &'static str {
        self.transitions[seed as usize % self.transitions.len()]
    }

    /// Returns a conclusion starter phrase selected deterministically from `seed`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::phrases_mixer::PhrasesMixer;
    ///
    /// let phrase = PhrasesMixer::global().conclusion(1);
    /// assert!(!phrase.is_empty());
    /// ```
    pub fn conclusion(&self, seed: u64) -> &'static str {
        self.conclusions[seed as usize % self.conclusions.len()]
    }

    /// Returns an epistemic hedge phrase selected deterministically from `seed`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::phrases_mixer::PhrasesMixer;
    ///
    /// let phrase = PhrasesMixer::global().hedge(5);
    /// assert!(!phrase.is_empty());
    /// ```
    pub fn hedge(&self, seed: u64) -> &'static str {
        self.hedges[seed as usize % self.hedges.len()]
    }

    /// Enriches a paragraph by inserting a discourse marker before its second
    /// sentence when the paragraph contains at least two sentences.
    ///
    /// The insertion point and marker are chosen deterministically via `seed`
    /// so that repeated calls with the same seed produce identical output.
    ///
    /// # Arguments
    ///
    /// * `text` - Source paragraph text.
    /// * `seed` - Deterministic selection seed (typically the pipeline run seed).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::phrases_mixer::PhrasesMixer;
    ///
    /// let text = "Rust is fast. It is also memory-safe.";
    /// let enriched = PhrasesMixer::global().enrich_paragraph(text, 0);
    /// assert!(enriched.len() >= text.len());
    /// ```
    pub fn enrich_paragraph(&self, text: &str, seed: u64) -> String {
        let sentences: Vec<&str> = text.split(". ").collect();
        if sentences.len() < 2 {
            return text.to_string();
        }
        let marker = self.marker(seed);
        let mut parts = Vec::with_capacity(sentences.len() + 1);
        parts.push(sentences[0].to_string());
        parts.push(format!("{} {}", marker, sentences[1]));
        for s in &sentences[2..] {
            parts.push((*s).to_string());
        }
        parts.join(". ")
    }

    /// Prepends an academic opener to `text` when `text` does not already
    /// start with one of the openers in the bank.
    ///
    /// The trailing phrase ends with a space so that the original text flows
    /// on naturally: `"<Opener> <text>"`.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to potentially wrap with an opener.
    /// * `seed` - Deterministic selection seed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::generation::phrases_mixer::PhrasesMixer;
    ///
    /// let plain = "rust provides memory safety without a garbage collector.";
    /// let wrapped = PhrasesMixer::global().apply_opener_if_needed(plain, 2);
    /// // The wrapped text starts with an opener phrase.
    /// assert!(wrapped.len() > plain.len());
    /// ```
    pub fn apply_opener_if_needed(&self, text: &str, seed: u64) -> String {
        let lower = text.to_ascii_lowercase();
        let already_has_opener = self
            .openers
            .iter()
            .any(|op| lower.starts_with(&op.to_ascii_lowercase()));
        if already_has_opener {
            return text.to_string();
        }
        let opener = self.opener(seed);
        let trimmed = text.trim_start_matches(|c: char| c.is_uppercase());
        let first_char = text.chars().next().unwrap_or_default();
        let lowered_start = format!(
            "{}{}",
            first_char.to_ascii_lowercase(),
            &text[first_char.len_utf8()..]
        );
        let body = if trimmed.is_empty() {
            text
        } else {
            &lowered_start
        };
        format!("{} {}", opener, body)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
