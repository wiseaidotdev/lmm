// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `SearchOracle` - DuckDuckGo query planner with in-process cache.
//!
//! Transforms a query string into a raw-text observation that the ThinkLoop
//! controller feeds back as the error measurement.
//!
//! ## Retrieval Strategy
//!
//! Two DuckDuckGo APIs are tried in order, using only the `duckduckgo` crate:
//!
//! 1. **Instant Answer API** (`api.duckduckgo.com`) via `Browser::get_api_response`,
//!    returns a Wikipedia-quality `abstract_text` paragraph and related-topic
//!    sentences. This is the primary and most content-rich source.
//! 2. **Lite search** (`lite.duckduckgo.com`) via `Browser::lite_search`,
//!    HTML scraping fallback; produces title + snippet pairs that are less
//!    reliable but covers queries that the Instant Answer API misses.
//!
//! * **In-process cache**: duplicate queries never make a second network request.
//! * **Graceful offline degradation**: when `net` feature is disabled, `fetch`
//!   returns an empty string immediately.
//!
//! ## Examples
//!
//! ```rust
//! #[tokio::main]
//! async fn main() {
//!     use lmm::cognition::search::SearchOracle;
//!     let mut oracle = SearchOracle::new(5);
//!     let result = oracle.fetch("Rust memory safety").await;
//!     println!("{result}");
//! }
//! ```
//!
//! ## See Also
//!
//! * [Information retrieval - Wikipedia](https://en.wikipedia.org/wiki/Information_retrieval)
//! * [DuckDuckGo Instant Answer API](https://duckduckgo.com/duckduckgo-help-pages/open-source/instant-answer-interface/)

use std::collections::HashMap;

#[cfg(any(feature = "net", feature = "wasm-net"))]
use crate::constants::cognition_search::MAX_RELATED_TOPICS_SEARCH;
#[cfg(feature = "net")]
use crate::constants::net::{DEFAULT_BROWSER_AGENT, DEFAULT_USER_AGENT};
#[cfg(any(feature = "net", feature = "wasm-net"))]
use duckduckgo::browser::Browser;
#[cfg(feature = "net")]
use duckduckgo::user_agents::get as get_ua;

/// DuckDuckGo search client with an in-process query cache.
///
/// Queries are issued against two DuckDuckGo backends in priority order:
/// the Instant Answer API (rich Wikipedia summaries) and Lite search (HTML
/// scraped snippets). Results are cached in-process so the ThinkLoop never
/// issues a duplicate network round-trip.
///
/// # Examples
///
/// ```rust
/// #[tokio::main]
/// async fn main() {
///     use lmm::cognition::search::SearchOracle;
///     let mut oracle = SearchOracle::new(3);
///     let obs = oracle.fetch("Rust ownership").await;
///     println!("{obs}");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SearchOracle {
    /// In-process cache: query → observation corpus.
    cache: HashMap<String, String>,
    /// Maximum number of Lite result snippets to include (Instant Answer is unbounded).
    pub limit: usize,
    /// When true, disables all network requests (for WASM environments where CORS blocks requests).
    no_web: bool,
}

impl SearchOracle {
    /// Constructs a new `SearchOracle`.
    ///
    /// * `limit`: maximum Lite-search snippets to concatenate (≥ 1).
    pub fn new(limit: usize) -> Self {
        Self {
            cache: HashMap::new(),
            limit: limit.max(1),
            no_web: false,
        }
    }

    /// Sets the no_web flag to disable all network requests.
    ///
    /// When true, `fetch()` will return empty strings without making any network calls.
    /// This is useful for WASM environments where CORS blocks cross-origin requests.
    pub fn no_web(mut self, no_web: bool) -> Self {
        self.no_web = no_web;
        self
    }

    /// Fetches an observation for `query`, returning a cached result when available.
    ///
    /// When the `net` feature is disabled this always returns `String::new()`.
    /// When `no_web` is true, this returns `String::new()` without any network calls.
    /// Network errors are silently discarded so the ThinkLoop continues safely.
    pub async fn fetch(&mut self, query: &str) -> String {
        if self.no_web {
            return String::new();
        }
        if let Some(cached) = self.cache.get(query) {
            return cached.clone();
        }
        let observation = self.fetch_live(query).await;
        self.cache.insert(query.to_string(), observation.clone());
        observation
    }

    /// Returns the number of cached queries.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Clears the in-process cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Seeds the in-process cache with a pre-fetched observation for `query`.
    ///
    /// This avoids redundant network round-trips when the caller has already
    /// fetched content for a topic and wants the ATE or ThinkLoop oracle to
    /// reuse it.  Only inserts if the query is not already cached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::search::SearchOracle;
    /// let mut oracle = SearchOracle::new(5);
    /// oracle.seed_cache("rust", "Rust is a general-purpose systems programming language.");
    /// ```
    pub fn seed_cache(&mut self, query: &str, observation: &str) {
        self.cache
            .entry(query.to_string())
            .or_insert_with(|| observation.to_string());
    }

    /// Returns the current no_web flag value.
    pub fn is_no_web(&self) -> bool {
        self.no_web
    }

    /// Issues the actual DuckDuckGo request (only compiled when `net` or `wasm-net` is on).
    ///
    /// Strategy:
    /// 1. Try `get_api_response` (Instant Answer API) → `abstract_text` + related topics.
    /// 2. If that yields nothing, try `lite_search` → title + snippet pairs.
    ///    (Lite search is skipped in WASM builds as it is blocked by CORS.)
    ///
    /// # Time Complexity
    ///
    /// O(1) network round-trips; O(|result|) string concatenation.
    #[cfg(any(feature = "net", feature = "wasm-net"))]
    async fn fetch_live(&self, query: &str) -> String {
        let browser = Browser::new();

        if let Some(text) = instant_answer(&browser, query).await {
            return text;
        }

        #[cfg(feature = "net")]
        {
            let ua = get_ua(DEFAULT_BROWSER_AGENT).unwrap_or(DEFAULT_USER_AGENT);
            return lite_snippets(&browser, query, ua, self.limit).await;
        }

        #[cfg(not(feature = "net"))]
        String::new()
    }

    #[cfg(not(any(feature = "net", feature = "wasm-net")))]
    async fn fetch_live(&self, _query: &str) -> String {
        String::new()
    }
}

/// Encodes a query string for use in a URL path: spaces become `+`, other
/// non-alphanumeric ASCII characters are percent-encoded.
///
/// # Time Complexity
///
/// O(|query|)
#[cfg(any(feature = "net", feature = "wasm-net"))]
fn url_encode_query(query: &str) -> String {
    query
        .chars()
        .map(|c| match c {
            ' ' => "+".to_string(),
            c if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' => c.to_string(),
            c => format!("%{:02X}", c as u32),
        })
        .collect()
}

/// Calls the DuckDuckGo Instant Answer API and assembles a text corpus from
/// `abstract_text`, `definition`, `answer`, and the first few `related_topics`.
///
/// Returns `None` when the response carries no usable text.
///
/// In WASM builds, `get_api_response` automatically falls back through public
/// CORS proxies and finally the Wikipedia API, so this always has a network
/// source available.
///
/// # Time Complexity
///
/// O(|related_topics|): linear over the topic list.
#[cfg(any(feature = "net", feature = "wasm-net"))]
async fn instant_answer(browser: &Browser, query: &str) -> Option<String> {
    let encoded = url_encode_query(query);
    let path = format!("?q={}&ia=web", encoded);
    let resp = browser.get_api_response(&path, None).await.ok()?;

    let mut parts: Vec<String> = Vec::new();

    if let Some(t) = resp.abstract_text.filter(|s| !s.trim().is_empty()) {
        parts.push(t);
    }
    if let Some(d) = resp.definition.filter(|s| !s.trim().is_empty()) {
        parts.push(d);
    }
    if let Some(a) = resp.answer.filter(|s| !s.trim().is_empty()) {
        parts.push(a);
    }

    for topic in resp.related_topics.iter().take(MAX_RELATED_TOPICS_SEARCH) {
        if let Some(text) = topic.text.as_deref().filter(|s| !s.trim().is_empty()) {
            parts.push(text.to_string());
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Calls DuckDuckGo Lite and assembles title + snippet pairs into a corpus.
///
/// Returns an empty string on error or when no results are returned.
///
/// **Not available in WASM** (`lite.duckduckgo.com` returns CORS errors in browsers).
///
/// # Time Complexity
///
/// O(|limit|): bounded by the requested result count.
#[cfg(feature = "net")]
async fn lite_snippets(browser: &Browser, query: &str, ua: &str, limit: usize) -> String {
    match browser.lite_search(query, "wt-wt", Some(limit), ua).await {
        Ok(results) => results
            .iter()
            .filter_map(|r| {
                let snippet = r.snippet.trim();
                let title = r.title.trim();
                if !snippet.is_empty() {
                    Some(format!("{} {}", title, snippet))
                } else if !title.is_empty() {
                    Some(title.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        Err(_) => String::new(),
    }
}
