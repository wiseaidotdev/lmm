// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # Stochastic Constants

pub const DEFAULT_REPLACEMENT_PROBABILITY: f64 = 0.22;

pub const PROBABILITY_MIN: f64 = 0.0;

pub const PROBABILITY_MAX: f64 = 1.0;

/// Rate at which random entropy (novel word swaps) is injected into the
/// stochastic enhancement pass, independent of the synonym replacement
/// probability.
pub const ENTROPY_INJECTION_RATE: f64 = 0.35;

/// Maximum number of recursive phrase substitution passes applied to a
/// single sentence during stochastic enhancement.
pub const PHRASE_SUBSTITUTION_DEPTH: usize = 3;

/// When `false`, the system word-list fallback (which picks a same-length
/// word from `/usr/share/dict/words`) is disabled.  Only curated thesaurus
/// synonyms are used, preventing nonsense substitutions such as
/// `"programming"` → `"algonquian"`.
pub const ENABLE_WORDLIST_FALLBACK: bool = false;

#[cfg(not(target_arch = "wasm32"))]
pub const WORDLIST_MIN_WORD_LEN: usize = 5;

#[cfg(not(target_arch = "wasm32"))]
pub const WORDLIST_MAX_WORD_LEN: usize = 14;

#[cfg(not(target_arch = "wasm32"))]
pub static SYSTEM_DICT_PATHS: &[&str] = &[
    "/usr/share/dict/american-english",
    "/usr/share/dict/english",
    "/usr/share/dict/words",
    "/usr/dict/words",
];

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.
