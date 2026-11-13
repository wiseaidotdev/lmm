// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # TRAM Scraper Constants

pub const MIN_SENTENCE_LENGTH: usize = 8;
pub const MAX_PARAGRAPH_LENGTH: usize = 500;
pub const MIN_ENGLISH_RATIO: f64 = 0.7;

pub const MIN_DEEP_PARAGRAPH_WORDS: usize = 35;
pub const MIN_THINK_DEEP_PARAGRAPH_WORDS: usize = 25;
pub const MAX_DEEP_PARAGRAPHS_TO_SHOW: usize = 5;
pub const MAX_LINKED_URLS_TO_FOLLOW: usize = 5;

pub const EMPTY_GAP_PATTERN: &str = " , ";
pub const ARTICLE_GAP_PATTERNS: &[&str] = &[" the .", " a .", " an .", " the ,", " a ,", " an ,"];
pub const INCOMPLETE_PARENTHETICAL_PATTERNS: &[&str] = &[" (e.g.,", " (i.e.,"];
pub const INCOMPLETE_QUOTE_PATTERN: &str = " from \"";
pub const INCOMPLETE_QUOTE_EXCEPTION: &str = " from \"mostly";

pub const CITATION_SEMICOLON_GAP_PATTERN: &str = " or ;";
pub const CITATION_SEMICOLON_AND_GAP_PATTERN: &str = " and ;";
pub const CITATION_SEMICOLON_WAS_GAP_PATTERN: &str = " was ;";

pub const INCOMPLETE_THROUGH_PATTERN: &str = " designed to be highly through";
pub const INCOMPLETE_THROUGH_MODULES_PATTERN: &str = " highly through modules";
pub const INCOMPLETE_COMMON_PATTERN: &str = "a common in the";

pub const CODE_KEYWORDS: &[&str] = &[
    "block expression",
    "semicolon",
    "fn main()",
    "println!",
    "let mut",
];
pub const CODE_INDICATORS: &[&str] = &["`", "pure }", "typestates"];
pub const TABLE_INDICATORS: &[&str] = &["│", "|"];

pub const SENTENCE_END_INCOMPLETE: &[&str] = &[" the ", " of ", " for "];
pub const STANDALONE_PUNCTUATION: &[&str] = &["}", ")"];

pub const MONTH_PATTERNS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];
