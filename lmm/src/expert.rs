// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Expert System Module
//!
//! This module implements an expert-system framework for TRAM.
//!
//! ## Sub-modules
//!
//! | Module | Contents |
//! |---|---|
//! | [`rule`]        | IF-THEN production rules, `Condition`, `RuleBase` |
//! | [`inference`]   | Backward-chaining `InferenceEngine`, `InferenceResult` |
//! | [`uncertainty`] | Prospector certainty-factor model, `CertaintyFactor` |
//! | [`explain`]     | How/why explanation facility, `ExplanationTrace`, `ProofStep` |

pub mod explain;
pub mod inference;
pub mod rule;
pub mod uncertainty;

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
