// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Skill Constants

/// Directory (relative to the user's home or working directory) where skill
/// JSON files are stored on disk.
pub const SKILL_STORE_DIR: &str = ".lmm/skills";

/// File extension appended to every persisted skill file.
pub const SKILL_FILE_EXT: &str = ".skill.json";

/// Minimum confidence score a [`crate::cognition::skill::Skill`] must have to
/// be retained when the registry is loaded from disk.
pub const SKILL_MIN_CONFIDENCE: f64 = 0.4;

/// Maximum age in days before a skill is considered stale and dropped on load.
pub const SKILL_MAX_AGE_DAYS: u64 = 90;

/// Maximum number of skills held in the in-process hot cache.
pub const SKILL_HOT_CACHE_CAPACITY: usize = 32;

/// Label attached to knowledge sources that originate from a learned skill.
pub const SKILL_SOURCE_LABEL: &str = "learned-skill";

/// Version number assigned to a freshly created skill before any updates.
pub const SKILL_VERSION_INITIAL: u32 = 1;

/// Maximum byte length of a skill topic string.
pub const SKILL_TOPIC_MAX_LEN: usize = 128;

/// Subdirectory name used when exporting skills to a shared location.
pub const SKILL_SHARED_SKILLS_DIR: &str = "shared-skills";

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
