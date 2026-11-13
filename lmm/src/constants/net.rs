// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Network Constants

pub const DEFAULT_USER_AGENT: &str = "Mozilla/5.0";
#[cfg(feature = "net")]
pub const DEFAULT_BROWSER_AGENT: &str = "firefox";
pub const MIN_WORDS_FOR_TOPIC_PREFIX: usize = 5;
pub const MIN_WORDS_FOR_TITLE: usize = 3;
pub const MIN_WORDS_FOR_SNIPPET: usize = 7;
pub const MAX_RELATED_TOPICS: usize = 15;
pub const MAX_SOURCE_SNIPPETS: usize = 6;
pub const MAX_TOPIC_WORDS: usize = 6;
