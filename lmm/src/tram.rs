// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM (Text Reasoning and Applied Mathematics)
//!
//! Core TRAM functionality for programmatic use. This module provides
//! reusable components for query derivation, response quality filtering,
//! web knowledge retrieval, and the complete TRAM reasoning pipeline.
//!
//! The `net` feature enables all web functionality on both native and WASM
//! targets. On `wasm32` the pipeline uses the Wikipedia REST API (via
//! `gloo-net`) for deep corpus extraction instead of the native HTML scraper.
//!
//! ## Example
//!
//! ```rust,no_run
//! use lmm::tram::{TramConfig, TramPipeline};
//! use lmm::cognition::knowledge::KnowledgeIndex;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = TramConfig::builder()
//!     .iterations(10)
//!     .inference_depth(5)
//!     .think(true)
//!     // .deep(true)
//!     .deep(true)
//!     .build();
//!
//! let mut index = KnowledgeIndex::new();
//! let pipeline = TramPipeline::new(config);
//! let result = pipeline.run("What is Rust programming language?", &mut index).await?;
//!
//! println!("{}", result.final_text);
//! # Ok(())
//! # }
//! ```

pub mod cache;
pub mod config;
pub mod deep_scrape;
pub mod intent_filter;
pub mod pipeline;
pub mod query;
pub mod response;
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
pub mod scraper;
pub mod web_fetch;

pub use config::WebFetchResult;
pub use config::{
    DEFAULT_CF_THRESHOLD, DEFAULT_MIN_RULE_COUNT, TramConfig, TramConfigBuilder, TramResult,
};
pub use intent_filter::IntentFilter;
pub use pipeline::TramPipeline;
pub use query::{
    extract_creative_subject, extract_topic, split_sentences_with_punct, strip_discourse_connector,
    strip_leading_articles, strip_trailing_verb, trim_predicate,
};
pub use response::{
    assemble_paragraphs, calculate_relevance_score, deduplicate_sentences,
    enhance_with_prose_connectors, enhance_with_shaper_connectors, filter_sentences,
    frame_response, is_quality_answer, is_valid_summary, restructure_analogical_response,
};
#[cfg(all(feature = "net", target_arch = "wasm32"))]
pub use web_fetch::fetch_wikipedia_corpus;
#[cfg(feature = "net")]
pub use web_fetch::{prefetch_web, prefetch_web_for_query};

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
