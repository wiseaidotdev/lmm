// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to these terms.

//! # Encode Constants

pub const REPLACEMENT_CHAR: char = '\u{FFFD}';

pub const EQUATION_VARIABLE_NAME: &str = "x";
pub const DEFAULT_EVALUATION_FALLBACK: f64 = 65.0;
pub const EMPTY_TEXT_ERROR: &str = "Cannot encode empty text";
pub const RESIDUAL_SEPARATOR: char = ',';
pub const DEFAULT_RESIDUAL: i64 = 0;
pub const UNICODE_CODE_POINT_MIN: i64 = 0;
pub const UNICODE_CODE_POINT_MAX: i64 = 0x10FFFF;
