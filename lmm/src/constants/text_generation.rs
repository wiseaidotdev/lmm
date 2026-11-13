// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # Text Generation Constants

pub const DEFAULT_TOPIC: &str = "reality";
pub const DEFAULT_EMPTY_STRING: &str = "";
pub const ARTICLE_THE: &str = "the";
pub const KEYWORD_COUNT: usize = 3;
pub const VARIANT_COUNT: usize = 6;
pub const TONE_OFFSET_STEP: f64 = 1.0;
pub const TONE_OFFSET_ADJ_NEG: f64 = 2.0;
pub const TONE_OFFSET_ADJ_POS: f64 = 3.0;
pub const TONE_OFFSET_OBJ_NEG: f64 = 2.0;
pub const TONE_OFFSET_OBJ_POS: f64 = 1.0;
pub const TONE_OFFSET_LINK_POS: f64 = 1.0;
pub const TONE_OFFSET_SUBJ_POS: f64 = 1.5;
pub const TONE_OFFSET_ADJ_MAX: f64 = 3.0;
pub const TONE_OFFSET_VARIANT_3_VERB: f64 = 2.0;
pub const TONE_OFFSET_VARIANT_4_ADJ: f64 = 1.0;
pub const TONE_OFFSET_VARIANT_4_SUBJ: f64 = 1.0;
pub const TONE_OFFSET_VARIANT_4_OBJ: f64 = 2.0;
pub const TONE_OFFSET_VARIANT_5_SUBJ: f64 = 3.0;
pub const TONE_OFFSET_VARIANT_5_VERB: f64 = 1.5;
pub const TONE_OFFSET_VARIANT_5_ADJ: f64 = 3.0;
pub const DEFAULT_TONE: f64 = 110.0;

pub const VARIANT_INDEX_OFFSET_1: usize = 1;
pub const VARIANT_INDEX_OFFSET_2: usize = 2;
pub const VARIANT_INDEX_OFFSET_3: usize = 3;
pub const VARIANT_INDEX_OFFSET_4: usize = 4;
pub const VARIANT_INDEX_OFFSET_5: usize = 5;
pub const TONE_OFFSET_STEP_DBL: f64 = 2.0;
pub const MIN_WORDS_FOR_SENTENCE: usize = 5;
pub const DEDUP_WORD_LIMIT: usize = 6;
pub const TITLE_WORD_LIMIT: usize = 7;
