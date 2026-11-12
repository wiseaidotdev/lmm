// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod generation;
pub mod topic;

pub use generation::{EssayGenerator, ParagraphGenerator, SentenceGenerator, TextSummarizer};
pub use topic::{aspect_topic, summarize_to_topic};
