// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Answer Formatting Constants
//!
//! Constants for answer formatting, prefix generation, and query type detection.

pub const ADVANTAGES_PREFIX_TEMPLATE: &str = "The advantages of {} are as follows. {}";
pub const BENEFITS_PREFIX_TEMPLATE: &str = "The benefits of {} are as follows. {}";

pub const ADVANTAGE_KEYWORDS: &[&str] = &["advantage", "benefit", "pro", "strength"];
pub const HISTORY_KEYWORDS: &[&str] =
    &["history", "when", "created", "founded", "began", "started"];
pub const HISTORICAL_MARKERS: &[&str] = &[
    "created",
    "founded",
    "sponsored",
    "release",
    "developer",
    "foundation",
    "began",
    "started",
];

pub const MIN_FILTERED_PARAGRAPHS: usize = 1;
pub const MIN_DEEP_PARAGRAPHS: usize = 3;
