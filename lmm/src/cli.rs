// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # CLI Module
//!
//! Command-line interface modules including display, logging, utilities,
//! TRAM pipeline logic, and command handlers.

pub mod commands;
pub mod display;
pub mod handlers;
pub mod logging;
pub mod tram;
pub mod util;

pub use commands::Cli;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
