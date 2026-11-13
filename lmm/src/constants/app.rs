// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to these terms.

//! # App Constants

pub const HISTORICAL_MARKERS: &[&str] = &[
    "created",
    "founded",
    "sponsored",
    "release",
    "developer",
    "foundation",
];

pub const HISTORY_PROMPT_KEYWORDS: &[&str] = &["history", "when", "created", "founded"];

pub const EVALUATIVE_KEYWORDS: &[&str] = &[
    "advantage",
    "benefit",
    "pros",
    "disadvantage",
    "inconvenience",
    "limitation",
    "drawback",
    "weakness",
    "cons",
];

pub const VS_SPLITTER_PRIMARY: &str = " vs ";
pub const VS_SPLITTER_SECONDARY: &str = " versus ";

pub const DEFAULT_CODE_LANGUAGE: &str = "text";

pub const ESSAY_LABEL_INTRODUCTION: &str = "Introduction";
pub const ESSAY_LABEL_CONCLUSION: &str = "Conclusion";
pub const ESSAY_LABEL_BODY_PREFIX: &str = "Body · §";

pub const DEFAULT_STOCHASTIC_PROBABILITY: f64 = 0.12;
pub const PROBABILITY_MIN: f64 = 0.0;
pub const PROBABILITY_MAX: f64 = 1.0;
pub const MIN_EXPERT_PROSE_WORDS: usize = 30;
