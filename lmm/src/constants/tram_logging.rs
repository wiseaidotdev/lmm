// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Logging Constants
//!
//! Constants for logging message templates.

pub const WEB_SEARCH_LOG_PREFIX: &str = "  🔍 Web Search: querying DDG for: ";
pub const DEEP_SEARCH_LOG_PREFIX: &str = "  🔍 Deep Search: tried query='";
pub const DEEP_SEARCH_FOUND_PREFIX: &str = "', found ";
pub const DEEP_SEARCH_URLS_PREFIX: &str = " URLs: ";
pub const IA_FOUND_LOG_PREFIX: &str = "  🔍 Web Search: IA abstract found (";
pub const IA_FOUND_LOG_SUFFIX: &str = " chars)";
pub const DEEP_SCRAPE_LOG_PREFIX: &str = "  🔍 Deep Search: found ";
pub const DEEP_SCRAPE_LOG_MID: &str = " relevant URLs from ";
pub const DEEP_SCRAPE_LOG_SUFFIX: &str = " iterations";
