// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Pipeline CLI Modules
//!
//! CLI-specific adapters for TRAM functionality. This module re-exports
//! core TRAM functions and adds CLI-specific display and logging wrappers.

pub mod pipeline;
pub mod query;
pub mod response;

pub use pipeline::{prefetch_web, prefetch_web_for_query, run_tram_pipeline};
pub use query::{extract_topic, split_sentences_with_punct, strip_discourse_connector};
pub use response::{
    assemble_paragraphs_for, derive_framing_topic, restructure_analogical_response,
};

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
