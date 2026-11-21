// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Learning Engine Persistence Constants

/// Filename used to persist the [`LearningEngine`][crate::cognition::learning::engine::LearningEngine]
/// state to disk between TRAM pipeline runs.
///
/// The file is written as JSON using `serde_json` and stored in the same
/// directory as the skill registry (the platform data directory joined with
/// [`SKILL_STORE_DIR`][crate::constants::skill::SKILL_STORE_DIR]).
pub const LEARNING_ENGINE_STATE_FILE: &str = "lmm_learning_state.json";

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
