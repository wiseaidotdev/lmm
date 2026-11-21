// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Response, Re-export Hub
//!

pub use crate::cli::tram::framing::derive_framing_topic;
pub use crate::cli::tram::kb_query::derive_kb_query;
pub use crate::cli::tram::paragraph::{assemble_paragraphs_for, restructure_analogical_response};
pub use crate::cli::tram::quality::{is_answer_intent_relevant, is_quality_answer};
pub use crate::cli::tram::symbolic::build_symbolic_response;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
