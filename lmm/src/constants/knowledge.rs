// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Knowledge Constants

pub const DEFAULT_EMPTY_EXTENSION: &str = "";
pub const DEFAULT_FILE_LABEL: &str = "file";

/// Source label used when ingesting verbatim code blocks via [`KnowledgeIndex::ingest_code_block`].
///
/// All chunks stored under this label bypass the minimum-word-count filter so that
/// short snippets such as `print("Hello, World!")` are retained in the index.
/// [`KnowledgeIndex::query_code_blocks`] returns every chunk under this label directly
/// without semantic ranking, ensuring code blocks are always available to [`CodeResponder`].
pub const CODE_BLOCK_SOURCE_LABEL: &str = "code-blocks";

/// Minimum whitespace-delimited word count for a sentence chunk to be ingested.
pub const MIN_SENTENCE_CHUNK_WORDS: usize = 4;

/// Minimum whitespace-delimited word count for a paragraph chunk to be ingested.
pub const MIN_PARAGRAPH_CHUNK_WORDS: usize = 8;

/// Delimiter used to split text into paragraph-level blocks.
pub const PARAGRAPH_SEPARATOR: &str = "\n\n";

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
