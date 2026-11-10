// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Novel Prose Generation Module
//!
//! This module provides higher-level text synthesis for TRAM's expert path,
//! going beyond extractive retrieval and synonym substitution to build
//! **novel, coherent paragraphs** grounded in expert inference results.
//!
//! ## Sub-modules
//!
//! | Module | Contents |
//! |---|---|
//! | [`prose`]     | `ProseBuilder`: slot-filling paragraph synthesis |
//! | [`coherence`] | `CoherenceScorer`: adjacent-sentence reordering |
//! | [`filter`]    | `SentenceFilter`: declarative noise-rejection gate |
//! | [`dedup`]     | `GlobalDeduplicator`: Jaccard-based near-duplicate removal |
//! | [`shaper`]    | `ParagraphShaper`: structural paragraph constraint enforcer |

pub mod coherence;
pub mod dedup;
pub mod filter;
pub mod prose;
pub mod shaper;

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
