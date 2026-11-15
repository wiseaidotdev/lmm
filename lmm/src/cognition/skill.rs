// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Skill Registry: on-the-fly skill learning and retrieval.
//!
//! This module provides two public types:
//!
//! * [`Skill`]: a single learned capability keyed by a natural-language topic.
//! * [`SkillRegistry`]: an in-process hot cache backed by a directory of JSON
//!   files, with fuzzy topic matching, confidence filtering, and age-based
//!   expiry.
//!
//! Skills are persisted as `<topic-slug>.skill.json` files inside a configurable
//! directory.  On load, stale (too old) or low-confidence skills are silently
//! dropped.

use crate::constants::skill::{
    SKILL_FILE_EXT, SKILL_MAX_AGE_DAYS, SKILL_MIN_CONFIDENCE, SKILL_SHARED_SKILLS_DIR,
    SKILL_TOPIC_MAX_LEN,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A single learned skill: a topic paired with structured content, a
/// confidence score, a creation timestamp, and a schema version.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::skill::Skill;
///
/// let skill = Skill {
///     topic: "rust ownership".to_string(),
///     content: "Rust uses ownership to manage memory without a GC.".to_string(),
///     confidence: 0.9,
///     created_at: 0,
///     version: 1,
/// };
/// assert_eq!(skill.topic, "rust ownership");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Natural-language label for this skill (e.g. `"rust ownership"`).
    pub topic: String,
    /// Structured textual content describing the skill.
    pub content: String,
    /// Confidence score in the range `[0.0, 1.0]`.
    pub confidence: f64,
    /// Unix timestamp (seconds since epoch) when the skill was first created.
    pub created_at: u64,
    /// Schema version; starts at [`crate::constants::skill::SKILL_VERSION_INITIAL`].
    pub version: u32,
}

/// In-process skill registry backed by a JSON-file store.
///
/// Skills are kept in a `HashMap` for O(1) exact-topic lookups and O(n) fuzzy
/// scans.  The registry is loaded eagerly from disk on construction and written
/// back with [`SkillRegistry::flush`].
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::skill::{Skill, SkillRegistry};
///
/// let dir = std::env::temp_dir().join("skill_registry_doctest");
/// let mut reg = SkillRegistry::load_or_create(&dir);
/// let skill = Skill {
///     topic: "closures".to_string(),
///     content: "Closures capture their environment.".to_string(),
///     confidence: 0.8,
///     created_at: 0,
///     version: 1,
/// };
/// assert!(reg.upsert(skill));
/// assert!(reg.get("closures").is_some());
/// ```
#[derive(Debug, Default)]
pub struct SkillRegistry {
    store_path: PathBuf,
    hot_skills: HashMap<String, Skill>,
}

impl SkillRegistry {
    /// Constructs a registry by scanning `path` for `*.skill.json` files.
    ///
    /// Skills whose confidence is below [`SKILL_MIN_CONFIDENCE`] or whose age
    /// exceeds [`SKILL_MAX_AGE_DAYS`] are silently skipped.  If `path` does
    /// not exist it is created.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::skill::SkillRegistry;
    ///
    /// let dir = std::env::temp_dir().join("skill_load_or_create_doctest");
    /// let reg = SkillRegistry::load_or_create(&dir);
    /// assert!(reg.is_empty());
    /// ```
    pub fn load_or_create(path: impl Into<PathBuf>) -> Self {
        let store_path: PathBuf = path.into();
        let _ = std::fs::create_dir_all(&store_path);

        let now = now_secs();
        let max_age_secs = SKILL_MAX_AGE_DAYS * 86_400;

        let mut hot_skills = HashMap::new();

        if let Ok(entries) = std::fs::read_dir(&store_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str())
                    != Some(SKILL_FILE_EXT.trim_start_matches('.'))
                {
                    continue;
                }
                let ext_check = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.ends_with(SKILL_FILE_EXT))
                    .unwrap_or(false);
                if !ext_check {
                    continue;
                }
                if let Ok(json) = std::fs::read_to_string(&p)
                    && let Ok(skill) = serde_json::from_str::<Skill>(&json)
                {
                    if skill.confidence < SKILL_MIN_CONFIDENCE {
                        continue;
                    }
                    if now.saturating_sub(skill.created_at) > max_age_secs {
                        continue;
                    }
                    hot_skills.insert(skill.topic.clone(), skill);
                }
            }
        }

        Self {
            store_path,
            hot_skills,
        }
    }

    /// Returns a reference to the skill whose topic matches `topic`.
    ///
    /// An exact (case-sensitive) lookup is tried first; if that fails a fuzzy
    /// [`Self::topic_matches`] scan is performed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::skill::{Skill, SkillRegistry};
    ///
    /// let dir = std::env::temp_dir().join("skill_get_doctest");
    /// let mut reg = SkillRegistry::load_or_create(&dir);
    /// let skill = Skill {
    ///     topic: "iterators".to_string(),
    ///     content: "Iterators are lazy in Rust.".to_string(),
    ///     confidence: 0.85,
    ///     created_at: 0,
    ///     version: 1,
    /// };
    /// reg.upsert(skill);
    /// assert!(reg.get("iterators").is_some());
    /// assert!(reg.get("iterator").is_some());
    /// ```
    pub fn get(&self, topic: &str) -> Option<&Skill> {
        if let Some(skill) = self.hot_skills.get(topic) {
            return Some(skill);
        }
        self.hot_skills
            .iter()
            .find(|(stored, _)| Self::topic_matches(stored, topic))
            .map(|(_, skill)| skill)
    }

    /// Inserts or updates a skill in the hot cache.
    ///
    /// Returns `true` when the skill is brand-new (no prior entry with the
    /// same topic existed), and `false` when an existing entry was updated.
    ///
    /// Topics longer than [`SKILL_TOPIC_MAX_LEN`] are silently truncated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::skill::{Skill, SkillRegistry};
    ///
    /// let dir = std::env::temp_dir().join("skill_upsert_doctest");
    /// let mut reg = SkillRegistry::load_or_create(&dir);
    /// let skill = Skill {
    ///     topic: "traits".to_string(),
    ///     content: "Traits define shared behaviour.".to_string(),
    ///     confidence: 0.75,
    ///     created_at: 0,
    ///     version: 1,
    /// };
    /// assert!(reg.upsert(skill.clone()));
    /// assert!(!reg.upsert(skill));
    /// ```
    pub fn upsert(&mut self, mut skill: Skill) -> bool {
        if skill.topic.len() > SKILL_TOPIC_MAX_LEN {
            skill.topic.truncate(SKILL_TOPIC_MAX_LEN);
        }
        let is_new = !self.hot_skills.contains_key(&skill.topic);
        self.hot_skills.insert(skill.topic.clone(), skill);
        is_new
    }

    /// Persists all hot-cache skills to `<store_path>/<slug>.skill.json` files.
    ///
    /// # Errors
    ///
    /// Returns the first I/O error encountered while writing any file.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::skill::{Skill, SkillRegistry};
    ///
    /// let dir = std::env::temp_dir().join("skill_flush_doctest");
    /// let mut reg = SkillRegistry::load_or_create(&dir);
    /// let skill = Skill {
    ///     topic: "lifetimes".to_string(),
    ///     content: "Lifetimes prevent dangling references.".to_string(),
    ///     confidence: 0.9,
    ///     created_at: 0,
    ///     version: 1,
    /// };
    /// reg.upsert(skill);
    /// reg.flush().unwrap();
    /// ```
    pub fn flush(&self) -> std::io::Result<()> {
        let _ = std::fs::create_dir_all(&self.store_path);
        for skill in self.hot_skills.values() {
            let slug = topic_to_slug(&skill.topic);
            let mut filename = slug;
            filename.push_str(SKILL_FILE_EXT);
            let file_path = self.store_path.join(filename);
            let json = serde_json::to_string_pretty(skill)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            std::fs::write(file_path, json)?;
        }
        Ok(())
    }

    /// Copies all flushed skill files into `<target>/<SKILL_SHARED_SKILLS_DIR>/`.
    ///
    /// Returns the number of files successfully copied.
    ///
    /// # Errors
    ///
    /// Returns the first I/O error encountered during the copy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::skill::{Skill, SkillRegistry};
    ///
    /// let dir = std::env::temp_dir().join("skill_share_src_doctest");
    /// let target = std::env::temp_dir().join("skill_share_tgt_doctest");
    /// let mut reg = SkillRegistry::load_or_create(&dir);
    /// let skill = Skill {
    ///     topic: "enums".to_string(),
    ///     content: "Enums allow multiple variants.".to_string(),
    ///     confidence: 0.8,
    ///     created_at: 0,
    ///     version: 1,
    /// };
    /// reg.upsert(skill);
    /// reg.flush().unwrap();
    /// let count = reg.share_skills_to(&target).unwrap();
    /// assert_eq!(count, 1);
    /// ```
    pub fn share_skills_to(&self, target: &Path) -> std::io::Result<usize> {
        let shared_dir = target.join(SKILL_SHARED_SKILLS_DIR);
        std::fs::create_dir_all(&shared_dir)?;
        let mut copied = 0usize;
        if let Ok(entries) = std::fs::read_dir(&self.store_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                let is_skill_file = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.ends_with(SKILL_FILE_EXT))
                    .unwrap_or(false);
                if !is_skill_file {
                    continue;
                }
                if let Some(fname) = p.file_name() {
                    std::fs::copy(&p, shared_dir.join(fname))?;
                    copied += 1;
                }
            }
        }
        Ok(copied)
    }

    /// Returns `true` when at least half of the words in `query` are present
    /// (case-insensitive) in `stored`.
    ///
    /// Both strings are normalised to lowercase before splitting on whitespace.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::skill::SkillRegistry;
    ///
    /// assert!(SkillRegistry::topic_matches("rust programming", "rust program"));
    /// assert!(!SkillRegistry::topic_matches("python scripting", "rust ownership"));
    /// ```
    pub fn topic_matches(stored: &str, query: &str) -> bool {
        let stored_lower = stored.to_lowercase();
        let stored_words: Vec<&str> = stored_lower.split_whitespace().collect();
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        if query_words.is_empty() {
            return false;
        }

        let matches = query_words
            .iter()
            .filter(|qw| {
                stored_words
                    .iter()
                    .any(|sw| sw.starts_with(*qw) || *sw == **qw)
            })
            .count();

        matches * 2 >= query_words.len()
    }

    /// Returns the number of skills currently held in the hot cache.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::skill::SkillRegistry;
    ///
    /// let dir = std::env::temp_dir().join("skill_len_doctest");
    /// let reg = SkillRegistry::load_or_create(&dir);
    /// assert_eq!(reg.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.hot_skills.len()
    }

    /// Returns `true` when the hot cache contains no skills.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::skill::SkillRegistry;
    ///
    /// let dir = std::env::temp_dir().join("skill_is_empty_doctest");
    /// let reg = SkillRegistry::load_or_create(&dir);
    /// assert!(reg.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.hot_skills.is_empty()
    }
}

/// Converts a topic string into a filesystem-safe slug.
///
/// Spaces become underscores; non-alphanumeric, non-underscore characters are
/// stripped; the result is lowercased.
fn topic_to_slug(topic: &str) -> String {
    topic
        .to_lowercase()
        .replace(' ', "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Returns the current time as Unix seconds.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
