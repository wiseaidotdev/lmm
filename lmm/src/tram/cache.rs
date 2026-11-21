// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Corpus Cache
//!
//! Saves and loads the deep-scraped knowledge corpus for a given topic so that
//! subsequent runs skip the network round-trips and restore the KB directly from
//! disk.  Cache files live in `~/.lmm/cache/` (or the path set via
//! [`CORPUS_CACHE_ENV_VAR`]) as `{sanitized_topic}.json`.
//!
//! ## Format
//!
//! Each file is a JSON object with two keys:
//! - `"chunks"`: array of plain-text strings (prose and code-block chunks).
//! - `"urls"`: array of URL strings from the original deep-scrape.
//!
//! Chunk strings prefixed with [`CODE_BLOCK_MARKER`] are re-ingested via
//! [`KnowledgeIndex::ingest_code_block`]; all others via
//! [`KnowledgeIndex::ingest_text`] with source label [`CORPUS_CACHE_SOURCE_LABEL`].

use crate::cognition::knowledge::KnowledgeIndex;
use crate::constants::cache::{
    CORPUS_CACHE_ENV_VAR, CORPUS_CACHE_FALLBACK_DIR, CORPUS_CACHE_FILE_EXTENSION,
    CORPUS_CACHE_HOME_SUBDIR, CORPUS_CACHE_KEY_CHUNKS, CORPUS_CACHE_KEY_URLS,
    CORPUS_CACHE_MAX_AGE_SECS, CORPUS_CACHE_MAX_CHUNKS, CORPUS_CACHE_SEPARATOR,
    CORPUS_CACHE_SOURCE_LABEL, CORPUS_CACHE_STR_SEPARATOR, CORPUS_CACHE_SUBDIR,
};
use crate::constants::code_responder::CODE_BLOCK_MARKER;
use serde_json::{Value, from_str, to_string};
use std::fs::{create_dir_all, metadata, read_to_string, write};
use std::path::PathBuf;
use std::time::SystemTime;
use tracing::{info, warn};

/// Converts a topic string into a safe filesystem component.
///
/// Lowercases the string, replaces every non-alphanumeric character with
/// [`CORPUS_CACHE_SEPARATOR`], then collapses consecutive underscores and
/// trims leading/trailing ones.
///
/// # Examples
///
/// * `"hello world"` → `"hello_world"`
/// * `"integral of a number"` → `"integral_of_a_number"`
///
/// # Time Complexity
///
/// O(|topic|)
pub fn cache_key(topic: &str) -> String {
    let raw: String = topic
        .to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c
            } else {
                CORPUS_CACHE_SEPARATOR
            }
        })
        .collect();
    raw.split(CORPUS_CACHE_SEPARATOR)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(CORPUS_CACHE_STR_SEPARATOR)
}

/// Returns the filesystem path to the corpus cache directory.
///
/// Resolution order:
/// 1. `$LMM_DATA_DIR/cache` when the environment variable is set.
/// 2. `$HOME/.lmm/cache` when `$HOME` is available.
/// 3. `.lmm/cache` relative to the working directory as a last resort.
fn cache_dir() -> PathBuf {
    let base = std::env::var(CORPUS_CACHE_ENV_VAR).unwrap_or_else(|_| {
        std::env::var("HOME")
            .map(|h| format!("{}/{}", h, CORPUS_CACHE_HOME_SUBDIR))
            .unwrap_or_else(|_| CORPUS_CACHE_FALLBACK_DIR.to_string())
    });
    PathBuf::from(base).join(CORPUS_CACHE_SUBDIR)
}

/// Persists `chunks` and `urls` to disk under `{cache_dir}/{key}.json`.
///
/// At most [`CORPUS_CACHE_MAX_CHUNKS`] chunk entries are stored.
/// The file format is a JSON object with two keys: [`CORPUS_CACHE_KEY_CHUNKS`]
/// and [`CORPUS_CACHE_KEY_URLS`].
/// Any I/O errors are logged as warnings and silently swallowed so that a cache
/// write failure never aborts a query.
///
/// # Time Complexity
///
/// O(|chunks| + |urls|)
pub fn save_corpus(topic: &str, chunks: &[String], urls: &[String]) {
    let key = cache_key(topic);
    if key.is_empty() {
        return;
    }
    let dir = cache_dir();
    if let Err(e) = create_dir_all(&dir) {
        warn!("  💾 Cache: cannot create directory {:?}: {}", dir, e);
        return;
    }
    let path = dir.join(format!("{}.{}", key, CORPUS_CACHE_FILE_EXTENSION));
    let limited_chunks: Vec<&String> = chunks.iter().take(CORPUS_CACHE_MAX_CHUNKS).collect();
    let payload = serde_json::json!({
        CORPUS_CACHE_KEY_CHUNKS: limited_chunks,
        CORPUS_CACHE_KEY_URLS: urls,
    });
    match to_string(&payload) {
        Ok(json) => {
            if let Err(e) = write(&path, &json) {
                warn!("  💾 Cache: write failed for {:?}: {}", path, e);
            } else {
                info!(
                    "  💾 Cache SAVED: topic='{}' → {} chunks → {:?}",
                    topic,
                    limited_chunks.len(),
                    path
                );
            }
        }
        Err(e) => warn!("  💾 Cache: serialisation failed: {}", e),
    }
}

/// Loads a previously saved corpus from disk.
///
/// Returns `None` when no cache file exists or when the file is older than
/// [`CORPUS_CACHE_MAX_AGE_SECS`].  On load, chunks are re-ingested into
/// `index`: those prefixed with [`CODE_BLOCK_MARKER`] via
/// [`KnowledgeIndex::ingest_code_block`], the rest via
/// [`KnowledgeIndex::ingest_text`] with source [`CORPUS_CACHE_SOURCE_LABEL`].
///
/// Returns `Some(urls)` on a cache hit, where `urls` is the list of URLs
/// that were scraped during the original deep-search run.
///
/// # Arguments
///
/// * `topic`   - The topic whose cache to look up.
/// * `index`   - The knowledge index to populate when a cache hit occurs.
/// * `verbose` - Emit diagnostic log messages when `true`.
///
/// # Returns
///
/// `Some(Vec<String>)` on cache hit (index populated, URLs returned), `None` on miss.
///
/// # Time Complexity
///
/// O(|cached_chunks|)
pub fn load_corpus(topic: &str, index: &mut KnowledgeIndex, verbose: bool) -> Option<Vec<String>> {
    let key = cache_key(topic);
    if key.is_empty() {
        return None;
    }
    let dir = cache_dir();
    let path = dir.join(format!("{}.{}", key, CORPUS_CACHE_FILE_EXTENSION));
    if !path.exists() {
        return None;
    }
    let age_ok = metadata(&path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .map(|age| age.as_secs() <= CORPUS_CACHE_MAX_AGE_SECS)
        .unwrap_or(false);
    if !age_ok {
        if verbose {
            info!("  💾 Cache STALE: topic='{}' → re-scraping", topic);
        }
        return None;
    }
    let json = match read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            warn!("  💾 Cache: read failed for {:?}: {}", path, e);
            return None;
        }
    };
    let root: Value = match from_str(&json) {
        Ok(v) => v,
        Err(e) => {
            warn!("  💾 Cache: deserialisation failed: {}", e);
            return None;
        }
    };
    let chunks: Vec<String> = if root.is_array() {
        root.as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        root.get(CORPUS_CACHE_KEY_CHUNKS)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    };
    if chunks.is_empty() {
        return None;
    }
    let urls: Vec<String> = if root.is_array() {
        Vec::new()
    } else {
        root.get(CORPUS_CACHE_KEY_URLS)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut code_count = 0usize;
    let mut prose_count = 0usize;
    for chunk in &chunks {
        if chunk.starts_with(CODE_BLOCK_MARKER) {
            index.ingest_code_block(chunk);
            code_count += 1;
        } else {
            index.ingest_text(CORPUS_CACHE_SOURCE_LABEL, chunk);
            prose_count += 1;
        }
    }
    if verbose {
        info!(
            "  💾 Cache HIT: topic='{}' → {} prose + {} code-block chunk(s) restored from {:?}",
            topic, prose_count, code_count, path
        );
    }
    Some(urls)
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
