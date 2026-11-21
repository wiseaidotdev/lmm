// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Web Fetch Operations
//!
//! Unified web knowledge retrieval for both native and WASM targets under the
//! single `net` feature flag.
//!
//! ## Target-aware dispatch
//!
//! | Target   | DDG backend               | Deep corpus              |
//! |----------|---------------------------|--------------------------|
//! | native   | `Browser` (reqwest)       | HTML scraper + html2text |
//! | wasm32   | `SearchAggregator` (DDG)  | Wikipedia REST API       |
//!
//! The Wikipedia REST API (`en.wikipedia.org`) responds with
//! `Access-Control-Allow-Origin: *`, making it safe for browser fetch.
//! Code blocks are extracted and prefixed with [`CODE_BLOCK_MARKER`] so
//! [`CodeResponder`] treats them as high-confidence verified snippets,
//! identical to what the native HTML scraper produces.

use crate::cognition::knowledge::KnowledgeIndex;
#[cfg(feature = "net")]
use crate::constants::tram_web_fetch::{
    ADVANTAGES_PREFIX, BENEFITS_PREFIX, EVALUATIVE_KEYWORDS, PROS_CONS_PREFIX, WEB_IA_SOURCE,
    WHAT_IS_PREFIX,
};
use crate::tram::config::WebFetchResult;

#[cfg(all(feature = "net", target_arch = "wasm32"))]
use crate::constants::tram_web_fetch::{
    CANONICAL_HELLO_WORLD, KNOWN_PROGRAMMING_LANGUAGES, MIN_CODE_SNIPPET_CHARS,
    PL_DISAMBIGUATION_SUFFIX, WASM_CODE_EXAMPLE_QUERY, WASM_FAMOUS_POEMS_PREFIX,
    WASM_HOW_TO_IN_CONNECTOR, WASM_HOW_TO_WRITE_PREFIX, WASM_POEM_ABOUT_PREFIX,
    WIKIPEDIA_ALT_SOURCE, WIKIPEDIA_FETCH_QUERY_PREFIX, WIKIPEDIA_MOBILE_SECTIONS_URL,
    WIKIPEDIA_OPENSEARCH_URL, WIKIPEDIA_PL_DISAMBIG, WIKIPEDIA_QUERY_SOURCE, WIKIPEDIA_SOURCE,
};

#[cfg(all(feature = "net", target_arch = "wasm32"))]
use crate::cognition::task_decomposer::{CreativeOutputKind, TaskDecomposer};

#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use crate::constants::tram_web_fetch::{
    ADVANTAGES_OF_PREFIX, AND_CONNECTOR, BENEFITS_OF_PREFIX, CAUSAL_KEYWORDS, CAUSAL_OPENERS,
    COMPARE_PREFIX, DIFFERENCES_BETWEEN_PREFIX, DISADVANTAGE_KEYWORDS, DISADVANTAGES_PREFIX,
    DUCKY_PREFIX, EMPTY_STRING, MAX_DEEP_SCRAPE_ITERATIONS, MAX_DUCKY_RESULTS, MAX_LITE_RESULTS,
    MAX_RELATED_TOPICS, MAX_TOPIC_PAIR_WORDS, MIN_ENTITY_WORD_LENGTH, MIN_IA_FALLBACK_WORD_COUNT,
    MIN_IA_WORD_COUNT, MIN_LITE_SNIPPET_WORD_COUNT, MIN_PROVERB_WORD_COUNT, OWNERSHIP_MODEL_SUFFIX,
    PROGRAMMING_LANGUAGE_SUFFIX, PROS_CONS_SUFFIX, VS_CONNECTOR, VS_SPLITTERS, WEB_DUCKY_SOURCE,
    WEB_IA_A_SOURCE, WEB_IA_B_SOURCE, WEB_IA_EVAL_SOURCE, WEB_IA_FALLBACK_SOURCE,
    WEB_INTENT_SOURCE, WEB_LITE_SOURCE, WEB_SCRAPE_SOURCE,
};

#[cfg(feature = "net")]
use crate::tram::query::{extract_creative_subject, extract_topic};

#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use crate::{
    cognition::thought::classifier::{InputClassifier, ReasoningMode},
    constants::{
        net::DEFAULT_BROWSER_AGENT,
        tram_logging::{IA_FOUND_LOG_PREFIX, IA_FOUND_LOG_SUFFIX, WEB_SEARCH_LOG_PREFIX},
    },
    net::{Browser, agent, user_agent_by_seed},
    tram::{
        cache::{load_corpus, save_corpus},
        deep_scrape::deep_scrape_for_topic,
        scraper::scrape_and_ingest,
    },
};

#[cfg(all(feature = "net", target_arch = "wasm32"))]
use crate::{
    constants::code_responder::CODE_BLOCK_MARKER,
    net::{Response, SearchAggregator, corpus_from_response, url_encode},
};
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use tracing::info;

#[cfg(all(feature = "net", target_arch = "wasm32"))]
use gloo_net::http::Request;

#[cfg(all(feature = "net", target_arch = "wasm32"))]
use serde::Deserialize;

/// Deserialized Wikipedia OpenSearch response tuple.
///
/// Format: `[query, [titles], [descriptions], [urls]]`
#[cfg(all(feature = "net", target_arch = "wasm32"))]
#[derive(Deserialize)]
struct WikiOpenSearch((), Vec<String>, (), ());

/// Strips all HTML tags from `html` using a minimal char-by-char scan.
///
/// Avoids pulling in a DOM parser, keeping the WASM binary compact.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
fn strip_html_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut inside = false;
    for ch in html.chars() {
        match ch {
            '<' => inside = true,
            '>' => inside = false,
            _ if !inside => out.push(ch),
            _ => {}
        }
    }
    out
}

/// Decodes the five standard HTML character entities in `text`.
///
/// Wikipedia HTML uses `&quot;`, `&amp;`, `&lt;`, `&gt;`, and `&#39;` inside
/// `<pre>`/`<code>` blocks.  Without decoding, extracted code is not runnable
/// (e.g. `print(&quot;Hello, World!&quot;)` instead of `print("Hello, World!")`).
#[cfg(all(feature = "net", target_arch = "wasm32"))]
fn decode_html_entities(text: &str) -> String {
    text.replace("&quot;", "\"")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

/// Extracts `<pre>` and `<code>` block text from `html` as a list of decoded
/// code snippets, each ready to be passed to [`KnowledgeIndex::ingest_code_block`].
///
/// Unlike the previous `String`-returning version, this function returns a
/// [`Vec<String>`] so callers can store each snippet via `ingest_code_block`,
/// bypassing the minimum-word-count filter that would otherwise drop short
/// one-liner snippets such as `print("Hello, World!")`.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
fn extract_code_snippets(html: &str) -> Vec<String> {
    let mut result = Vec::new();
    for tag in &["pre", "code"] {
        let open = format!("<{}", tag);
        let close = format!("</{}>", tag);
        let mut search = html;
        while let Some(start) = search.find(&open) {
            let after_open = &search[start..];
            if let Some(content_start) = after_open.find('>') {
                let content_area = &after_open[content_start + 1..];
                if let Some(end) = content_area.find(&close) {
                    let decoded = decode_html_entities(&strip_html_tags(&content_area[..end]));
                    let snippet = decoded.trim().to_string();
                    if snippet.len() > MIN_CODE_SNIPPET_CHARS {
                        result.push(snippet);
                    }
                    search = &content_area[end + close.len()..];
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
    result
}

/// Strips HTML from a section and joins non-empty lines into clean prose.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
fn html_to_prose(html: &str) -> String {
    strip_html_tags(html)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Fetches a Wikipedia article corpus for `query`, ingests code blocks directly
/// into `index`, and returns the prose text for the caller to ingest via
/// [`KnowledgeIndex::ingest_text`].
///
/// Code snippets extracted from `<pre>` and `<code>` tags are stored with
/// [`KnowledgeIndex::ingest_code_block`], which bypasses the minimum-word-count
/// filter — ensuring short snippets such as `print("Hello, World!")` survive
/// ingestion.  HTML character entities are decoded before storage so the
/// retrieved code is executable as-is.
///
/// Returns an empty `String` on any network or parse failure, allowing the
/// caller to degrade gracefully.
///
/// ## Steps
///
/// 1. **OpenSearch** - resolve `query` to the best matching article title.
/// 2. **mobile-sections** - fetch lead + remaining section HTML from Wikipedia.
/// 3. **Extract** - strip HTML, decode entities, store code blocks, join prose.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
pub async fn fetch_wikipedia_corpus(query: &str, index: &mut KnowledgeIndex) -> String {
    let search_url = format!("{}{}", WIKIPEDIA_OPENSEARCH_URL, url_encode(query));

    let title = match Request::get(&search_url).send().await {
        Ok(r) if r.ok() => match r.json::<WikiOpenSearch>().await {
            Ok(WikiOpenSearch(_, titles, _, _)) => titles.into_iter().next().unwrap_or_default(),
            Err(_) => return String::new(),
        },
        _ => return String::new(),
    };

    if title.is_empty() {
        return String::new();
    }

    let sections_url = format!("{}{}", WIKIPEDIA_MOBILE_SECTIONS_URL, url_encode(&title));

    let json = match Request::get(&sections_url).send().await {
        Ok(r) if r.ok() => match r.json::<serde_json::Value>().await {
            Ok(v) => v,
            Err(_) => return String::new(),
        },
        _ => return String::new(),
    };

    let mut prose = String::new();

    let mut ingest_sections = |sections: &serde_json::Value| {
        if let Some(arr) = sections.as_array() {
            for section in arr {
                if let Some(html) = section.get("text").and_then(|v| v.as_str()) {
                    for snippet in extract_code_snippets(html) {
                        index.ingest_code_block(&format!("{}{}", CODE_BLOCK_MARKER, snippet));
                    }
                    prose.push_str(&html_to_prose(html));
                    prose.push(' ');
                }
            }
        }
    };

    if let Some(sections) = json.get("lead").and_then(|l| l.get("sections")) {
        ingest_sections(sections);
    }
    if let Some(sections) = json.get("remaining").and_then(|r| r.get("sections")) {
        ingest_sections(sections);
    }

    prose
}

/// Builds a Wikipedia OpenSearch query optimised for code examples.
///
/// Returns the exact Wikipedia article title most likely to contain verified
/// code snippets for the given `language` and `subject`.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
fn wikipedia_code_query(language: &str, subject: &str) -> String {
    let lower = subject.to_ascii_lowercase();
    if lower.contains("hello") || lower.contains("world") {
        "Hello, World! program".to_string()
    } else if lower.contains("factorial") {
        "Factorial".to_string()
    } else if lower.contains("fibonacci") {
        "Fibonacci sequence".to_string()
    } else if lower.contains("sort") || lower.contains("sorting") {
        format!("{} sort algorithm", language)
    } else if lower.contains("binary") && lower.contains("search") {
        "Binary search algorithm".to_string()
    } else if lower.contains("linked list") {
        "Linked list".to_string()
    } else if lower.contains("binary tree") || lower.contains("tree") {
        "Binary tree".to_string()
    } else if lower.contains("prime") {
        "Sieve of Eratosthenes".to_string()
    } else {
        format!("{} (programming language)", language)
    }
}

/// Collects the abstract URL and up to five related-topic URLs from a DDG
/// [`Response`] into `out`, de-duplicating by value.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
fn collect_ddg_urls(resp: &Response, out: &mut Vec<String>) {
    if let Some(url) = resp.abstract_url.as_deref() {
        if !url.is_empty() && !out.contains(&url.to_string()) {
            out.push(url.to_string());
        }
    }
    for topic in resp.related_topics.iter().take(5) {
        if let Some(url) = topic.first_url.as_deref() {
            if !url.is_empty() && !out.contains(&url.to_string()) {
                out.push(url.to_string());
            }
        }
    }
}

/// Returns the canonical Hello-World snippet for `language` from the
/// compile-time [`CANONICAL_HELLO_WORLD`] table, or `None` when the
/// language is not listed.
///
/// Used as a local fallback when Wikipedia code extraction yields no blocks,
/// ensuring `CodeResponder` always has a verified snippet to present for the
/// most common programming languages without requiring a network round-trip.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
fn canonical_hello_world_snippet(language: &str) -> Option<&'static str> {
    let key = language.to_ascii_lowercase();
    CANONICAL_HELLO_WORLD
        .iter()
        .find(|(lang, _)| *lang == key.as_str())
        .map(|(_, snippet)| *snippet)
}

/// WASM implementation of `prefetch_web_for_query`.
///
/// Issues a DDG Instant Answer query and falls back to Wikipedia when the
/// response corpus is empty.  Code snippets extracted from Wikipedia `<pre>`
/// and `<code>` tags are stored via [`KnowledgeIndex::ingest_code_block`] so
/// they bypass the minimum-word-count filter.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
async fn prefetch_web_for_query_wasm(query: &str, index: &mut KnowledgeIndex) -> WebFetchResult {
    let aggregator = SearchAggregator::new();
    let mut ingested = 0usize;

    if let Ok(resp) = aggregator.get_response(query).await {
        let corpus = corpus_from_response(&resp);
        if !corpus.is_empty() {
            ingested += index.ingest_text(WEB_IA_SOURCE, &corpus);
        }
    }

    if ingested == 0 {
        let wiki_prose = fetch_wikipedia_corpus(query, index).await;
        if !wiki_prose.is_empty() {
            ingested += index.ingest_text(WIKIPEDIA_QUERY_SOURCE, &wiki_prose);
        }
    }

    WebFetchResult {
        total_chunks_ingested: ingested,
        prose_chunks_ingested: ingested,
        fetch_queries: vec![query.to_string()],
        ..Default::default()
    }
}

/// WASM implementation of `prefetch_web`.
///
/// Mirrors the full CLI `prefetch_web` strategy for browser targets:
///
/// 1. **DDG Instant Answer** - focused queries matching the prompt intent.
///    Code prompts issue targeted example/how-to queries. Evaluative prompts
///    issue advantages/benefits/pros-cons queries. Generic prompts issue topic
///    overview queries. The list is kept short to minimise duplicate ingestion.
/// 2. **Wikipedia code corpus** - for code prompts, fetches the Wikipedia
///    article most likely to contain verified `<pre>`/`<code>` blocks (e.g.
///    `"Hello, World! program"`) plus the language reference article.  Code
///    snippets are stored via [`KnowledgeIndex::ingest_code_block`], bypassing
///    the minimum-word-count filter; prose goes through `ingest_text`.
/// 3. **Wikipedia topic corpus** - for evaluative/factual prompts with deep
///    mode, fetches the primary topic article for prose enrichment.
#[cfg(all(feature = "net", target_arch = "wasm32"))]
async fn prefetch_web_wasm(
    prompt: &str,
    index: &mut KnowledgeIndex,
    _kb_provided: bool,
    deep: bool,
) -> WebFetchResult {
    let topic = extract_topic(prompt);
    let creative_kind = TaskDecomposer::detect_creative_kind(prompt);
    let aggregator = SearchAggregator::new();
    let prompt_lower = prompt.to_ascii_lowercase();

    let is_evaluative = EVALUATIVE_KEYWORDS
        .iter()
        .any(|&kw| prompt_lower.contains(kw));

    let queries: Vec<String> = if let CreativeOutputKind::Code { ref language } = creative_kind {
        let subject = extract_creative_subject(prompt);
        vec![
            format!("{} {}{}", language, subject, WASM_CODE_EXAMPLE_QUERY),
            format!(
                "{}{}{}{}",
                WASM_HOW_TO_WRITE_PREFIX, subject, WASM_HOW_TO_IN_CONNECTOR, language
            ),
            language.clone(),
        ]
    } else if matches!(
        creative_kind,
        CreativeOutputKind::Poem | CreativeOutputKind::Story | CreativeOutputKind::Essay
    ) {
        let subject = extract_creative_subject(prompt);
        vec![
            format!("{}{}", WASM_POEM_ABOUT_PREFIX, subject),
            format!("{}{}", WASM_FAMOUS_POEMS_PREFIX, subject),
            subject.clone(),
        ]
    } else if is_evaluative {
        let topic_lc = topic.to_ascii_lowercase();
        let already_has_pl_suffix = topic_lc.contains(PL_DISAMBIGUATION_SUFFIX);
        let is_pl = already_has_pl_suffix
            || KNOWN_PROGRAMMING_LANGUAGES
                .iter()
                .any(|&pl| topic_lc == pl || topic_lc.starts_with(&format!("{} ", pl)));
        let effective_topic = if is_pl && !already_has_pl_suffix {
            format!("{} {}", topic, PL_DISAMBIGUATION_SUFFIX)
        } else {
            topic.clone()
        };
        vec![
            format!("{}{}", ADVANTAGES_PREFIX, effective_topic),
            format!("{}{}", BENEFITS_PREFIX, effective_topic),
            format!("{}{}", PROS_CONS_PREFIX, effective_topic),
            effective_topic,
        ]
    } else {
        vec![topic.clone(), format!("{}{}", WHAT_IS_PREFIX, topic)]
    };

    let mut total_chunks = 0usize;
    let mut fetch_queries = Vec::new();
    let mut all_urls: Vec<String> = Vec::new();

    for query in &queries {
        if let Ok(resp) = aggregator.get_response(query).await {
            let corpus = corpus_from_response(&resp);
            if !corpus.is_empty() {
                total_chunks += index.ingest_text(WEB_IA_SOURCE, &corpus);
            }
            collect_ddg_urls(&resp, &mut all_urls);
            fetch_queries.push(query.to_string());
        }
    }

    if deep || matches!(creative_kind, CreativeOutputKind::Code { .. }) {
        if let CreativeOutputKind::Code { ref language } = creative_kind {
            let subject = extract_creative_subject(prompt);
            let wiki_query = wikipedia_code_query(language, &subject);
            let wiki_prose = fetch_wikipedia_corpus(&wiki_query, index).await;
            if !wiki_prose.is_empty() {
                total_chunks += index.ingest_text(WIKIPEDIA_SOURCE, &wiki_prose);
                fetch_queries.push(wiki_query);
            }
            let lang_prose =
                fetch_wikipedia_corpus(&format!("{}{}", language, WIKIPEDIA_PL_DISAMBIG), index)
                    .await;
            if !lang_prose.is_empty() {
                total_chunks += index.ingest_text(WIKIPEDIA_ALT_SOURCE, &lang_prose);
            }
            if index.query_code_blocks().is_empty() {
                if let Some(snippet) = canonical_hello_world_snippet(language) {
                    index.ingest_code_block(&format!("{}{}", CODE_BLOCK_MARKER, snippet));
                    total_chunks += 1;
                }
            }
        } else if deep {
            let topic_lc = topic.to_ascii_lowercase();
            let already_has_pl_suffix = topic_lc.contains(PL_DISAMBIGUATION_SUFFIX);
            let is_pl = already_has_pl_suffix
                || KNOWN_PROGRAMMING_LANGUAGES
                    .iter()
                    .any(|&pl| topic_lc == pl || topic_lc.starts_with(&format!("{} ", pl)));
            let wiki_topic = if is_pl && is_evaluative {
                let base = if already_has_pl_suffix {
                    let words: Vec<&str> = topic.split_whitespace().collect();
                    words
                        .iter()
                        .take_while(|w| !w.to_ascii_lowercase().contains(PL_DISAMBIGUATION_SUFFIX))
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" ")
                } else {
                    topic.clone()
                };
                format!("{}{}", base.trim(), WIKIPEDIA_PL_DISAMBIG)
            } else {
                topic.clone()
            };
            let wiki_prose = fetch_wikipedia_corpus(&wiki_topic, index).await;
            if !wiki_prose.is_empty() {
                total_chunks += index.ingest_text(WIKIPEDIA_SOURCE, &wiki_prose);
                fetch_queries.push(format!("{}{}", WIKIPEDIA_FETCH_QUERY_PREFIX, wiki_topic));
            }
        }
    }

    WebFetchResult {
        total_chunks_ingested: total_chunks,
        prose_chunks_ingested: total_chunks,
        fetch_queries,
        all_scraped_urls: all_urls.clone(),
        successful_urls: all_urls,
        ..Default::default()
    }
}

/// Fires a lightweight intent-specific DDG search and ingests snippets into `index`.
///
/// Unlike `prefetch_web` this never skips on `kb_provided`; it is meant for
/// secondary enrichment queries generated by task decomposers.
///
/// On native targets uses the full DuckDuckGo `Browser` client with lite search
/// fallback and HTML scraping. On `wasm32` uses `SearchAggregator` and falls
/// back to the Wikipedia REST API.
///
/// # Arguments
///
/// * `query`   - The search query string.
/// * `index`   - The `KnowledgeIndex` to populate.
/// * `verbose` - When `true`, enables detailed debug logging (native only).
///
/// # Returns
///
/// A `WebFetchResult` containing ingestion statistics.
///
/// # Time Complexity
///
/// O(|network response|)
#[cfg(feature = "net")]
pub async fn prefetch_web_for_query(
    query: &str,
    index: &mut KnowledgeIndex,
    verbose: bool,
) -> WebFetchResult {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = verbose;
        return prefetch_web_for_query_wasm(query, index).await;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let browser = Browser::new();
        let ua = agent(DEFAULT_BROWSER_AGENT).unwrap_or_else(|| user_agent_by_seed(0));
        let encoded: String = query
            .chars()
            .map(|c| if c == ' ' { '+' } else { c })
            .collect();
        let ia_path = format!("?q={}&ia=web", encoded);
        let mut ingested = 0usize;
        let mut fetch_queries = Vec::new();

        if verbose {
            info!("{}{}", WEB_SEARCH_LOG_PREFIX, query);
        }
        if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
            if let Some(text) = resp.abstract_text.filter(|s| !s.trim().is_empty()) {
                if verbose {
                    info!(
                        "{}{}{}",
                        IA_FOUND_LOG_PREFIX,
                        text.len(),
                        IA_FOUND_LOG_SUFFIX
                    );
                }
                ingested += index.ingest_text(WEB_INTENT_SOURCE, text.trim());
            }
            if let Some(def) = resp.definition.filter(|s| !s.trim().is_empty()) {
                ingested += index.ingest_text(WEB_INTENT_SOURCE, def.trim());
            }
        }

        if ingested == 0 {
            if let Ok(results) = browser
                .lite_search(query, "wt-wt", Some(MAX_LITE_RESULTS), ua)
                .await
            {
                for r in &results {
                    let snippet = r.snippet.trim();
                    let title = r.title.trim();
                    if snippet.split_whitespace().count() >= MIN_PROVERB_WORD_COUNT {
                        let chunk = format!("{} {}", title, snippet);
                        ingested += index.ingest_text(WEB_INTENT_SOURCE, chunk.trim());
                    }
                }
                fetch_queries.push(query.to_string());
            }
        } else {
            fetch_queries.push(query.to_string());
        }

        if ingested == 0 {
            let ducky_q = format!("{}{}", DUCKY_PREFIX, query.replace(' ', "+"));
            if let Ok(ducky_results) = browser
                .lite_search(&ducky_q, "wt-wt", Some(MAX_DUCKY_RESULTS), ua)
                .await
            {
                for r in &ducky_results {
                    let snippet = r.snippet.trim();
                    let title = r.title.trim();
                    if snippet.split_whitespace().count() >= MIN_LITE_SNIPPET_WORD_COUNT {
                        let chunk = format!("{} {}", title, snippet);
                        ingested += index.ingest_text(WEB_DUCKY_SOURCE, chunk.trim());
                    }
                    let url = &r.url;
                    if url.starts_with("http") {
                        let scraped = scrape_and_ingest(url, index, WEB_SCRAPE_SOURCE).await;
                        if scraped > 0 {
                            ingested += scraped;
                        }
                    }
                }
                if ingested > 0 {
                    fetch_queries.push(ducky_q);
                }
            }
        }

        WebFetchResult {
            total_chunks_ingested: ingested,
            prose_chunks_ingested: 0,
            lite_snippets_found: 0,
            instant_answer_query: None,
            fetch_queries,
            unscraped_urls: Vec::new(),
            all_scraped_urls: Vec::new(),
            successful_urls: Vec::new(),
        }
    }
}

#[cfg(not(feature = "net"))]
pub async fn prefetch_web_for_query(
    _query: &str,
    _index: &mut KnowledgeIndex,
    _verbose: bool,
) -> WebFetchResult {
    WebFetchResult::default()
}

/// Fires a main DDG search with Instant Answer and Lite search, ingesting into `index`.
///
/// Skips entirely when `kb_provided` is true (local KB was supplied).
///
/// # Arguments
///
/// * `prompt`      - The search query to issue.
/// * `index`       - The `KnowledgeIndex` to populate.
/// * `kb_provided` - When `true`, a local KB was supplied and the web fetch is skipped.
/// * `deep`        - When `true`, enables deep scraping for topic-relevant content.
/// * `verbose`     - When `true`, enables detailed debug logging.
///
/// # Returns
///
/// A `WebFetchResult` containing ingestion statistics and query details.
///
/// # Time Complexity
///
/// O(|network response|)
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
pub async fn prefetch_web(
    prompt: &str,
    index: &mut KnowledgeIndex,
    kb_provided: bool,
    deep: bool,
    verbose: bool,
) -> WebFetchResult {
    if kb_provided {
        return WebFetchResult::default();
    }

    if deep {
        let mode = InputClassifier::classify(prompt);
        let early_topic = if matches!(mode, ReasoningMode::TaskExecution) {
            extract_creative_subject(prompt)
        } else {
            extract_topic(prompt)
        };
        if let Some(cached_urls) = load_corpus(&early_topic, index, verbose) {
            if verbose {
                info!(
                    "  💾 Deep Search: skipped (cache hit for '{}')",
                    early_topic
                );
            }
            return WebFetchResult {
                total_chunks_ingested: 0,
                prose_chunks_ingested: 0,
                lite_snippets_found: 0,
                instant_answer_query: None,
                fetch_queries: Vec::new(),
                unscraped_urls: Vec::new(),
                all_scraped_urls: Vec::new(),
                successful_urls: cached_urls,
            };
        }
    }

    let browser = Browser::new();
    let ua = agent(DEFAULT_BROWSER_AGENT).unwrap_or_else(|| user_agent_by_seed(0));
    let mut ingested = 0usize;
    let mut prose_chunks = 0usize;
    let mut lite_snippets_found = 0usize;
    let mut fetch_queries = Vec::new();

    let topic_query = extract_topic(prompt);
    let topic_clean = topic_query.trim().to_string();
    let topic_lang = format!("{}{}", topic_query, PROGRAMMING_LANGUAGE_SUFFIX);

    let topic_advantages = format!("{}{}", ADVANTAGES_PREFIX, topic_query);
    let topic_benefits = format!("{}{}", BENEFITS_PREFIX, topic_query);
    let topic_disadvantages = format!("{}{}", DISADVANTAGES_PREFIX, topic_query);
    let topic_pros_cons = format!("{}{}", PROS_CONS_PREFIX, topic_query);
    let topic_what = format!("{}{}", WHAT_IS_PREFIX, topic_query);

    let topic_words: Vec<String> = topic_query
        .split_whitespace()
        .filter(|w| w.len() >= MIN_ENTITY_WORD_LENGTH)
        .map(|w| w.to_string())
        .collect();

    let prompt_lower = prompt.to_ascii_lowercase();
    let is_evaluative_prompt = EVALUATIVE_KEYWORDS
        .iter()
        .any(|&keyword| prompt_lower.contains(keyword));
    let is_causal_prompt = prompt_lower.starts_with("why ")
        || prompt_lower.starts_with("how come")
        || CAUSAL_KEYWORDS
            .iter()
            .any(|&keyword| prompt_lower.contains(keyword));
    let is_analogical_prompt = topic_query.to_ascii_lowercase().contains(" vs ")
        || topic_query.to_ascii_lowercase().contains(" versus ")
        || prompt_lower.contains(" vs ")
        || prompt_lower.contains(" versus ");

    let causal_predicate_query: Option<String> = if is_causal_prompt {
        let lower_clean = prompt_lower
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim();
        let predicate = CAUSAL_OPENERS
            .iter()
            .find_map(|&pfx| lower_clean.strip_prefix(pfx))
            .unwrap_or(lower_clean)
            .trim();
        if predicate.len() > topic_query.len() + 2 {
            let words: Vec<&str> = predicate.split_whitespace().collect();
            let query_words = if words.len() > MAX_TOPIC_PAIR_WORDS {
                &words[..MAX_TOPIC_PAIR_WORDS]
            } else {
                &words
            };
            Some(query_words.join(" "))
        } else {
            None
        }
    } else {
        None
    };

    let (entity_lang_a, entity_lang_b) = if is_analogical_prompt {
        let tq_lower = topic_query.to_ascii_lowercase();
        let mut pair: (Option<String>, Option<String>) = (None, None);
        for &splitter in VS_SPLITTERS {
            if let Some(pos) = tq_lower.find(splitter) {
                let a = topic_query[..pos].trim().to_string();
                let b = topic_query[pos + splitter.len()..].trim().to_string();
                if !a.is_empty() && !b.is_empty() {
                    pair = (Some(a), Some(b));
                    break;
                }
            }
        }
        pair
    } else {
        (None, None)
    };

    let (entity_a, entity_b) = if is_analogical_prompt {
        let tq_lower = topic_query.to_ascii_lowercase();
        let mut pair: (Option<String>, Option<String>) = (None, None);
        for &splitter in VS_SPLITTERS {
            if let Some(pos) = tq_lower.find(splitter) {
                let a = topic_query[..pos].trim().to_string();
                let b = topic_query[pos + splitter.len()..].trim().to_string();
                if !a.is_empty() && !b.is_empty() {
                    pair = (Some(a), Some(b));
                    break;
                }
            }
        }
        pair
    } else {
        (None, None)
    };

    let analogical_entity_queries: Vec<String> = {
        let mut v = Vec::new();
        if let Some(ref ea) = entity_a {
            v.push(ea.clone());
        }
        if let Some(ref eb) = entity_b {
            v.push(eb.clone());
        }
        v
    };

    let is_evaluative_must_fetch = is_evaluative_prompt;

    let eval_queries: Vec<String> = if is_evaluative_prompt {
        let wants_disadvantages = DISADVANTAGE_KEYWORDS
            .iter()
            .any(|&keyword| prompt_lower.contains(keyword));
        if wants_disadvantages {
            vec![topic_disadvantages.clone(), topic_pros_cons.clone()]
        } else {
            vec![
                topic_advantages.clone(),
                topic_benefits.clone(),
                topic_pros_cons.clone(),
            ]
        }
    } else {
        vec![]
    };

    let must_fetch_queries: Vec<String> = {
        let mut v = analogical_entity_queries.clone();
        if let Some(ref cpq) = causal_predicate_query
            && !v.iter().any(|q| q == cpq)
        {
            v.push(cpq.clone());
        }
        v.extend(eval_queries.iter().cloned());
        v
    };

    let mut ia_candidates: Vec<String> = Vec::new();

    if is_causal_prompt && let Some(ref cpq) = causal_predicate_query {
        ia_candidates.push(cpq.clone());
        ia_candidates.push(format!("why {}", cpq));
        ia_candidates.push(format!("{} {}", topic_query, cpq));
    }

    if is_analogical_prompt {
        if let Some(ref ea) = entity_a {
            ia_candidates.push(ea.clone());
        }
        if let Some(ref eb) = entity_b {
            ia_candidates.push(eb.clone());
        }
    }

    if let Some(ref ea) = entity_lang_a {
        ia_candidates.push(ea.clone());
    }
    if let Some(ref eb) = entity_lang_b {
        ia_candidates.push(eb.clone());
    }

    ia_candidates.push(topic_query.clone());
    ia_candidates.push(topic_lang.clone());

    for tw in &topic_words {
        if !ia_candidates.iter().any(|q| q == tw) {
            ia_candidates.push(tw.clone());
        }
    }

    ia_candidates.push(topic_what.clone());
    ia_candidates.push(topic_clean.clone());
    ia_candidates.push(prompt.to_string());

    if is_evaluative_prompt {
        ia_candidates.extend(eval_queries.iter().cloned());
    }

    let mut ia_winner: Option<String> = None;

    for query in ia_candidates.iter() {
        let is_must_fetch = must_fetch_queries.iter().any(|q| q == query);
        let is_entity_query = is_analogical_prompt
            && (entity_a
                .as_ref()
                .map(|ea| query.contains(ea))
                .unwrap_or(false)
                || entity_b
                    .as_ref()
                    .map(|eb| query.contains(eb))
                    .unwrap_or(false));
        let is_evaluative_query = is_evaluative_must_fetch
            && (query.contains("advantages")
                || query.contains("benefits")
                || query.contains("disadvantages")
                || query.contains("pros"));
        if prose_chunks > 0 && !is_must_fetch && !is_entity_query && !is_evaluative_query {
            break;
        }
        let encoded = query
            .chars()
            .map(|c| if c == ' ' { '+' } else { c })
            .collect::<String>();
        let ia_path = format!("?q={}&ia=web", encoded);
        if verbose {
            info!("{}{}", WEB_SEARCH_LOG_PREFIX, query);
        }
        if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
            if let Some(text) = resp.abstract_text.filter(|s| !s.trim().is_empty()) {
                if verbose {
                    info!(
                        "{}{}{}",
                        IA_FOUND_LOG_PREFIX,
                        text.len(),
                        IA_FOUND_LOG_SUFFIX
                    );
                }
                let wc = text.split_whitespace().count();
                let added = index.ingest_text(WEB_IA_SOURCE, text.trim());
                if added > 0 && wc >= MIN_IA_WORD_COUNT {
                    prose_chunks += added;
                }
                ingested += added;
            }
            if let Some(def) = resp.definition.filter(|s| !s.trim().is_empty()) {
                let wc = def.split_whitespace().count();
                let added = index.ingest_text("web-ia", def.trim());
                if added > 0 && wc >= MIN_IA_WORD_COUNT {
                    prose_chunks += added;
                }
                ingested += added;
            }
            if let Some(ans) = resp.answer.filter(|s| !s.trim().is_empty()) {
                let wc = ans.split_whitespace().count();
                let added = index.ingest_text("web-ia", ans.trim());
                if added > 0 && wc >= MIN_IA_WORD_COUNT {
                    prose_chunks += added;
                }
                ingested += added;
            }
            for topic in resp.related_topics.iter().take(MAX_RELATED_TOPICS) {
                if let Some(text) = topic.text.as_deref().filter(|s| !s.trim().is_empty()) {
                    let wc = text.split_whitespace().count();
                    if wc >= MIN_IA_WORD_COUNT && text.split_whitespace().count() >= 8 {
                        let added = index.ingest_text(WEB_IA_SOURCE, text.trim());
                        prose_chunks += added;
                        ingested += added;
                    } else if wc >= MIN_PROVERB_WORD_COUNT && text.split_whitespace().count() >= 8 {
                        ingested += index.ingest_text("web-ia", text.trim());
                    }
                }
            }
            if ingested > 0 && ia_winner.is_none() {
                ia_winner = Some(query.clone());
            }
        }
    }

    if is_evaluative_must_fetch && prose_chunks == 0 {
        for eq in eval_queries {
            let encoded = eq.replace(' ', "+");
            let ia_path = format!("?q={}&ia=web", encoded);
            if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
                if let Some(text) = resp.abstract_text.filter(|s| !s.trim().is_empty()) {
                    let wc = text.split_whitespace().count();
                    let added = index.ingest_text(WEB_IA_EVAL_SOURCE, text.trim());
                    if added > 0 && wc >= MIN_IA_WORD_COUNT {
                        prose_chunks += added;
                    }
                    ingested += added;
                }
                if let Some(def) = resp.definition.filter(|s| !s.trim().is_empty()) {
                    let wc = def.split_whitespace().count();
                    let added = index.ingest_text("web-ia-eval", def.trim());
                    if added > 0 && wc >= MIN_IA_WORD_COUNT {
                        prose_chunks += added;
                    }
                    ingested += added;
                }
            }
        }
    }

    let needs_specific_content = is_evaluative_prompt || is_causal_prompt;
    let skip_lite = prose_chunks > 0 && !needs_specific_content;

    if is_analogical_prompt {
        if let Some(ref ea) = entity_a {
            let encoded_ea = ea.replace(' ', "+");
            let ia_path_ea = format!("?q={}&ia=web", encoded_ea);
            if let Ok(resp) = browser.get_api_response(&ia_path_ea, None).await {
                if let Some(text) = resp.abstract_text.filter(|s| !s.trim().is_empty()) {
                    let wc = text.split_whitespace().count();
                    let added = index.ingest_text(WEB_IA_A_SOURCE, text.trim());
                    if added > 0 && wc >= MIN_IA_WORD_COUNT {
                        prose_chunks += added;
                    }
                    ingested += added;
                }
                if let Some(def) = resp.definition.filter(|s| !s.trim().is_empty()) {
                    let wc = def.split_whitespace().count();
                    let added = index.ingest_text("web-ia-a", def.trim());
                    if added > 0 && wc >= MIN_IA_WORD_COUNT {
                        prose_chunks += added;
                    }
                    ingested += added;
                }
            }
        }

        if let Some(ref eb) = entity_b {
            let encoded_eb = eb.replace(' ', "+");
            let ia_path_eb = format!("?q={}&ia=web", encoded_eb);
            if let Ok(resp) = browser.get_api_response(&ia_path_eb, None).await {
                if let Some(text) = resp.abstract_text.filter(|s| !s.trim().is_empty()) {
                    let wc = text.split_whitespace().count();
                    let added = index.ingest_text(WEB_IA_B_SOURCE, text.trim());
                    if added > 0 && wc >= MIN_IA_WORD_COUNT {
                        prose_chunks += added;
                    }
                    ingested += added;
                }
                if let Some(def) = resp.definition.filter(|s| !s.trim().is_empty()) {
                    let wc = def.split_whitespace().count();
                    let added = index.ingest_text(WEB_IA_B_SOURCE, def.trim());
                    if added > 0 && wc >= MIN_IA_WORD_COUNT {
                        prose_chunks += added;
                    }
                    ingested += added;
                }
            }
        }
    }

    if !skip_lite {
        let mut lite_q_list: Vec<String> = Vec::new();
        let full_prompt_clean = prompt
            .trim()
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim()
            .to_string();
        if let Some(ref cpq) = causal_predicate_query {
            lite_q_list.push(cpq.clone());
            let topic_no_gc = format!("{} memory management", topic_query);
            lite_q_list.push(topic_no_gc);
            let topic_ownership = format!("{}{}", topic_query, OWNERSHIP_MODEL_SUFFIX);
            lite_q_list.push(topic_ownership);
        }
        if is_evaluative_prompt {
            lite_q_list.push(format!("{}{}", ADVANTAGES_OF_PREFIX, topic_query));
            lite_q_list.push(format!("{}{}", topic_query, PROS_CONS_SUFFIX));
            lite_q_list.push(format!("{}{}", BENEFITS_OF_PREFIX, topic_query));
        }
        if is_analogical_prompt {
            let ea_str = entity_a.as_deref().unwrap_or(EMPTY_STRING);
            let eb_str = entity_b.as_deref().unwrap_or(EMPTY_STRING);
            if let Some(ref ea) = entity_a {
                lite_q_list.push(ea.clone());
                lite_q_list.push(format!("{} vs {}", ea, eb_str));
            }
            if let Some(ref eb) = entity_b {
                lite_q_list.push(eb.clone());
                lite_q_list.push(format!("{}{}{}", ea_str, VS_CONNECTOR, eb));
            }
            lite_q_list.push(format!(
                "{}{}{}{}",
                COMPARE_PREFIX, ea_str, AND_CONNECTOR, eb_str
            ));
            lite_q_list.push(format!(
                "{}{}{}{}",
                DIFFERENCES_BETWEEN_PREFIX, ea_str, AND_CONNECTOR, eb_str
            ));
        }
        for eq in [entity_lang_a.as_deref(), entity_lang_b.as_deref()]
            .iter()
            .flatten()
        {
            if !lite_q_list.iter().any(|q| q.as_str() == *eq) {
                lite_q_list.push((*eq).to_string());
            }
        }

        lite_q_list.push(full_prompt_clean.clone());
        if topic_query.as_str() != prompt {
            lite_q_list.push(topic_query.clone());
        }
        if !lite_q_list.iter().any(|q| q.as_str() == prompt) {
            lite_q_list.push(prompt.to_string());
        }
        if !lite_q_list
            .iter()
            .any(|q| q.as_str() == topic_clean.as_str())
        {
            lite_q_list.push(topic_clean.clone());
        }
        let topic_words_significant: Vec<&str> = topic_query
            .split_whitespace()
            .filter(|w| {
                w.len() >= MIN_ENTITY_WORD_LENGTH
                    && !matches!(
                        w.to_ascii_lowercase().as_str(),
                        "what" | "does" | "have" | "that" | "this" | "with" | "from" | "into"
                    )
            })
            .collect();
        if topic_words_significant.len() >= 2 {
            for i in 0..topic_words_significant.len().saturating_sub(1) {
                let pair = format!(
                    "{} {}",
                    topic_words_significant[i],
                    topic_words_significant[i + 1]
                );
                if !lite_q_list.iter().any(|q| q == &pair) {
                    lite_q_list.push(pair);
                }
            }
        }
        for word in &topic_words_significant {
            if !lite_q_list.iter().any(|q| q.as_str() == *word) {
                lite_q_list.push((*word).to_string());
            }
        }

        'lite: for lite_q in &lite_q_list {
            match browser
                .lite_search(lite_q.as_str(), "wt-wt", Some(MAX_LITE_RESULTS), ua)
                .await
            {
                Ok(results) if !results.is_empty() => {
                    for r in &results {
                        let snippet = r.snippet.trim();
                        let title = r.title.trim();
                        if !snippet.is_empty() {
                            let chunk = format!("{} {}", title, snippet);
                            ingested += index.ingest_text(WEB_LITE_SOURCE, chunk.trim());
                            lite_snippets_found += 1;
                        }
                    }
                    if lite_snippets_found > 0 {
                        fetch_queries.push(lite_q.clone());
                        let was_supplementing = prose_chunks > 0;
                        if !was_supplementing {
                            break 'lite;
                        }
                    }
                }
                Ok(_) => {}
                Err(_e) => {
                    let encoded: String = lite_q
                        .chars()
                        .map(|c| if c == ' ' { '+' } else { c })
                        .collect();
                    let ia_path = format!("?q={}&ia=web", encoded);
                    if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
                        let mut ia_added = 0usize;
                        if let Some(text) = resp
                            .abstract_text
                            .filter(|s| s.split_whitespace().count() >= MIN_IA_FALLBACK_WORD_COUNT)
                        {
                            ia_added += index.ingest_text(WEB_IA_FALLBACK_SOURCE, text.trim());
                        }
                        if let Some(def) = resp
                            .definition
                            .filter(|s| s.split_whitespace().count() >= MIN_IA_FALLBACK_WORD_COUNT)
                        {
                            ia_added += index.ingest_text(WEB_IA_FALLBACK_SOURCE, def.trim());
                        }
                        if ia_added > 0 {
                            ingested += ia_added;
                            lite_snippets_found += ia_added;
                            fetch_queries.push(lite_q.clone());
                            let was_supplementing = prose_chunks > 0;
                            if !was_supplementing {
                                break 'lite;
                            }
                        } else {
                            break 'lite;
                        }
                    } else {
                        break 'lite;
                    }
                }
            }
        }
    }

    let mut unscraped_urls = Vec::new();
    let mut all_scraped_urls = Vec::new();
    let mut successful_urls = Vec::new();

    if deep {
        let mode = InputClassifier::classify(prompt);
        let topic = if matches!(mode, ReasoningMode::TaskExecution) {
            extract_creative_subject(prompt)
        } else {
            extract_topic(prompt)
        };

        let chunks_before = index.len();
        let deep_result =
            deep_scrape_for_topic(&topic, prompt, index, verbose, MAX_DEEP_SCRAPE_ITERATIONS).await;
        if deep_result.relevant_content_found > 0 {
            ingested += deep_result.total_chunks_ingested;
        } else if verbose {
            info!("  🔍 Deep Search: no relevant content found, skipping ingestion");
        }
        unscraped_urls = deep_result.unscraped_urls;
        all_scraped_urls = deep_result.all_scraped_urls;
        successful_urls = deep_result.successful_urls.clone();

        let new_chunks = index.all_chunk_texts_since(chunks_before);
        if !new_chunks.is_empty() {
            save_corpus(&topic, &new_chunks, &deep_result.successful_urls);
        }
    }

    WebFetchResult {
        total_chunks_ingested: ingested,
        prose_chunks_ingested: prose_chunks,
        lite_snippets_found,
        instant_answer_query: ia_winner,
        fetch_queries,
        unscraped_urls,
        all_scraped_urls,
        successful_urls,
    }
}

#[cfg(all(feature = "net", target_arch = "wasm32"))]
pub async fn prefetch_web(
    prompt: &str,
    index: &mut KnowledgeIndex,
    kb_provided: bool,
    deep: bool,
    _verbose: bool,
) -> WebFetchResult {
    prefetch_web_wasm(prompt, index, kb_provided, deep).await
}

#[cfg(not(feature = "net"))]
pub async fn prefetch_web(
    _prompt: &str,
    _index: &mut KnowledgeIndex,
    _kb_provided: bool,
    _deep: bool,
    _verbose: bool,
) -> WebFetchResult {
    WebFetchResult::default()
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
