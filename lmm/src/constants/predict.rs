// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to these terms.

//! # Predict Constants

pub const DEFAULT_WORD: &str = "the";
pub const DEFAULT_EVALUATION_FALLBACK: f64 = 107.0;
pub const TONE_DEVIATION_THRESHOLD: f64 = 6.0;
pub const TONE_MIN: f64 = 97.0;
pub const TONE_MAX: f64 = 122.0;
pub const LENGTH_MIN: f64 = 3.0;
pub const LENGTH_MAX: f64 = 10.0;
pub const DEFAULT_LENGTH: usize = 5;
pub const EQUATION_VARIABLE_NAME: &str = "x";
pub const MIN_SUFFIX_LENGTH: usize = 2;
pub const DEFAULT_FALLBACK_WORD: &str = "world";
pub const MIN_POSITIONS_FOR_TRAJECTORY: usize = 2;
pub const SYMBOLIC_REGRESSION_POPULATION_DEFAULT: usize = 60;
pub const SYMBOLIC_REGRESSION_POPULATION_RHYTHM: usize = 40;
pub const SYMBOLIC_REGRESSION_DEPTH_LIMIT: usize = 3;
pub const MIN_TOKENS_FOR_PREDICTION: usize = 2;
