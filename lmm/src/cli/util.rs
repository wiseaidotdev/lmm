// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # CLI Utilities
//!
//! Shared utility functions for CLI operations.

use crate::error::{LmmError, Result as LmmResult};
use std::fs;

/// Loads source text from a file path or uses the provided text.
///
/// # Arguments
///
/// * `input` - File path, or "-" to use the provided text.
/// * `text` - Text to use when input is "-".
///
/// # Time Complexity
///
/// O(|text|) when reading from memory; O(|file|) when reading from disk.
#[cfg(feature = "cli")]
pub fn load_source(input: &str, text: String) -> LmmResult<String> {
    if input == "-" {
        Ok(text)
    } else {
        fs::read_to_string(input)
            .map_err(|e| LmmError::Perception(e.to_string()))
            .map(|s| s.trim_end().to_string())
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
