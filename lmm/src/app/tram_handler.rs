// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Pipeline Handler (re-export)
//!
//! The monolithic `handle_tram` implementation has been decomposed into five
//! focused sub-stage modules under [`crate::app::tram`]:
//!
//! | Sub-module                           | Responsibility                              |
//! |--------------------------------------|---------------------------------------------|
//! | [`crate::app::tram::knowledge`]      | KB load · web pre-fetch · skill registry    |
//! | [`crate::app::tram::thinking`]       | ATE + ThinkLoop PID reasoner                |
//! | [`crate::app::tram::retrieval`]      | KB answer retrieval · deep-scrape filter    |
//! | [`crate::app::tram::assembly`]       | Expert inference · stochastic · framing     |
//! | [`crate::app::tram::rewards`]        | Episode reward · evaluative point synthesis |
//!
//! This module re-exports [`handle_tram`] for callers that still reference the
//! old path; all new code should import from `crate::app::tram` directly.

#[cfg(feature = "cli")]
pub use crate::app::tram::handle_tram;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
