// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Language Detection and Synonym Dispatch
//!
//! Provides [`LanguageHint`], a lightweight heuristic for detecting the
//! dominant script/language of a text snippet, and [`synonyms_for`], which
//! returns the appropriate compile-time PHF synonym table for the detected
//! language.
//!
//! ## Detection Algorithm
//!
//! The detector walks at most the first 400 bytes of the input and counts
//! codepoints that are characteristic of each supported script:
//!
//! | Language | Signal                                          |
//! |----------|-------------------------------------------------|
//! | Arabic   | Unicode Arabic block `U+0600-U+06FF`           |
//! | French   | `«»éèêëàâùûüîïç` or keywords `le/la/les/des` |
//! | German   | `äöüÄÖÜß` or keywords `der/die/das/und`        |
//! | Spanish  | `ñÑ¿¡` or keywords `es/las/los/una/del`        |
//! | English  | Fallback (pure ASCII or no strong signal)       |

use phf::Map;

use super::arabic::ARABIC_SYNONYMS;
use super::english::CURATED_SYNONYMS as ENGLISH_SYNONYMS;
use super::french::FRENCH_SYNONYMS;
use super::german::GERMAN_SYNONYMS;
use super::spanish::SPANISH_SYNONYMS;

/// The detected or requested dominant language of a text segment.
///
/// Used by [`SynonymBank`](super::SynonymBank) to route each lookup to the
/// correct compile-time PHF synonym table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LanguageHint {
    /// English (Moby Thesaurus II, ~25 k head words).
    #[default]
    English,
    /// Modern Standard Arabic script.
    Arabic,
    /// French.
    French,
    /// German.
    German,
    /// Spanish.
    Spanish,
}

/// Detects the dominant language of `text` using a lightweight heuristic.
///
/// Inspects at most 400 bytes of the input for script-specific codepoints
/// and high-frequency function words.  Falls back to [`LanguageHint::English`]
/// when no strong signal is found.
///
/// # Arguments
///
/// * `text`: Input text to analyse. May be any UTF-8 string.
///
/// # Returns
///
/// The [`LanguageHint`] with the strongest detected signal, or
/// [`LanguageHint::English`] on a tie.
///
/// # Time Complexity
///
/// O(min(|text|, 400)): bounded by the inspection window.
///
/// # Space Complexity
///
/// O(1): no allocations.
pub fn detect_language(text: &str) -> LanguageHint {
    let sample: &str = {
        let end = text
            .char_indices()
            .take(400)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(text.len());
        &text[..end]
    };

    let mut arabic_score: u32 = 0;
    let mut french_score: u32 = 0;
    let mut german_score: u32 = 0;
    let mut spanish_score: u32 = 0;

    for ch in sample.chars() {
        let cp = ch as u32;
        if (0x0600..=0x06FF).contains(&cp) {
            arabic_score += 2;
        }
        match ch {
            'é' | 'è' | 'ê' | 'ë' | 'à' | 'â' | 'ù' | 'û' | 'ü' | 'î' | 'ï' | 'ç' | '«' | '»' => {
                french_score += 1
            }
            'ä' | 'ö' | 'Ä' | 'Ö' | 'Ü' | 'ß' => german_score += 2,
            'ñ' | 'Ñ' | '¿' | '¡' => spanish_score += 2,
            _ => {}
        }
    }

    let lower = sample.to_ascii_lowercase();
    for word in lower.split_whitespace() {
        match word {
            "le" | "la" | "les" | "des" | "est" | "une" | "que" | "dans" | "sur" => {
                french_score += 1
            }
            "der" | "die" | "das" | "und" | "ist" | "von" | "mit" | "ein" | "eine" => {
                german_score += 1
            }
            "es" | "las" | "los" | "una" | "del" | "con" | "por" | "para" => spanish_score += 1,
            _ => {}
        }
    }

    let max_score = arabic_score
        .max(french_score)
        .max(german_score)
        .max(spanish_score);

    if max_score == 0 {
        return LanguageHint::English;
    }
    if arabic_score == max_score {
        return LanguageHint::Arabic;
    }
    if german_score == max_score {
        return LanguageHint::German;
    }
    if french_score == max_score {
        return LanguageHint::French;
    }
    if spanish_score == max_score {
        return LanguageHint::Spanish;
    }
    LanguageHint::English
}

/// Returns the compile-time PHF synonym map appropriate for `hint`.
///
/// # Arguments
///
/// * `hint`: The language to look up.
///
/// # Returns
///
/// A reference to one of the five static PHF maps.
///
/// # Time Complexity
///
/// O(1): simple pattern match.
///
/// # Space Complexity
///
/// O(1): returns a reference to static data.
pub(super) fn synonyms_for(
    hint: LanguageHint,
) -> &'static Map<&'static str, &'static [&'static str]> {
    match hint {
        LanguageHint::Arabic => &ARABIC_SYNONYMS,
        LanguageHint::French => &FRENCH_SYNONYMS,
        LanguageHint::German => &GERMAN_SYNONYMS,
        LanguageHint::Spanish => &SPANISH_SYNONYMS,
        LanguageHint::English => &ENGLISH_SYNONYMS,
    }
}
