// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Corpus Cache Constants
//!
//! Controls the on-disk cache that stores scraped knowledge-base chunks so that
//! repeated queries for the same topic skip the deep-scrape network round-trips.

/// Default base directory name under `$HOME` for the LMM data directory.
pub const CORPUS_CACHE_HOME_SUBDIR: &str = ".lmm";

/// Subdirectory under the LMM data dir for corpus cache files.
pub const CORPUS_CACHE_SUBDIR: &str = "cache";

/// Environment variable that overrides the entire LMM data base directory.
pub const CORPUS_CACHE_ENV_VAR: &str = "LMM_DATA_DIR";

/// Fallback base path when `$HOME` and `LMM_DATA_DIR` are both unset.
pub const CORPUS_CACHE_FALLBACK_DIR: &str = ".lmm/cache";

/// JSON key for the chunks array in the cache file format.
pub const CORPUS_CACHE_KEY_CHUNKS: &str = "chunks";

/// JSON key for the URLs array in the cache file format.
pub const CORPUS_CACHE_KEY_URLS: &str = "urls";

/// Maximum age of a cached corpus in seconds before it is considered stale and re-scraped.
///
/// Default: 7 days (604 800 s).  Stale files are ignored and replaced on the next run.
pub const CORPUS_CACHE_MAX_AGE_SECS: u64 = 60 * 60 * 24 * 7;

/// Maximum number of chunks stored per topic cache entry.
///
/// Keeps individual cache files from growing unbounded across many iterations.
pub const CORPUS_CACHE_MAX_CHUNKS: usize = 2000;

/// File extension used for cached corpus files on disk.
pub const CORPUS_CACHE_FILE_EXTENSION: &str = "json";

/// Source label assigned to chunks restored from the corpus cache.
pub const CORPUS_CACHE_SOURCE_LABEL: &str = "web-deep-scrape";

/// Character used to replace non-alphanumeric runes when building a cache key.
pub const CORPUS_CACHE_SEPARATOR: char = '_';

/// String separator used when joining sanitized cache key tokens.
pub const CORPUS_CACHE_STR_SEPARATOR: &str = "_";

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
