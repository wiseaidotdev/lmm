// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # Stochastic Constants

pub const DEFAULT_REPLACEMENT_PROBABILITY: f64 = 0.12;

pub const PROBABILITY_MIN: f64 = 0.0;

pub const PROBABILITY_MAX: f64 = 1.0;

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
