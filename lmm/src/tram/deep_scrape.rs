// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Deep Web Scraping for Topic-Relevant Content
//!
//! Iteratively scrapes websites discovered via DuckDuckGo Lite search to find
//! content similar to the topic being asked, and injects it into the knowledge
//! corpus. The scraping strategy uses a multi-tier URL collection approach:
//!
//! | Tier | Source | Description |
//! |------|--------|-------------|
//! | 1 | DDG Lite HTML search | Primary, yields real page URLs via POST |
//! | 2 | DDG IA `abstract_url` | Supplementary, Wikipedia-style entities only |
//! | 3 | DDG IA `related_topics` | Supplementary, related entity URLs |
//!
//! The previous implementation used only Tier 2 and 3, which returned 0 URLs
//! for contextual queries like "advantages of rust". Tier 1 is now the
//! primary source, resolving the 0-URL issue.

use crate::cognition::knowledge::KnowledgeIndex;
#[cfg(feature = "net")]
use crate::constants::tram_web_fetch::{
    DEEP_SCRAPE_BACK_OFF_BASE_MS, DEEP_SCRAPE_DDG_REDIRECT_PARAM, DEEP_SCRAPE_DOMAIN_FILTER,
    DEEP_SCRAPE_HTTP_PREFIX, DEEP_SCRAPE_IA_PARAM_WEB, DEEP_SCRAPE_KB_QUERY_LIMIT,
    DEEP_SCRAPE_KEYWORD_MIN_LENGTH, DEEP_SCRAPE_LITE_QUERY_REGION, DEEP_SCRAPE_MAX_RELATED_TOPICS,
    DEEP_SCRAPE_MAX_URLS_PER_ITERATION, DEEP_SCRAPE_RELEVANCE_KEYWORD_MIN_LENGTH,
    DEEP_SCRAPE_SOURCE_LABEL, MAX_DEEP_SCRAPE_ITERATIONS, MAX_DEEP_SCRAPE_LITE_RESULTS,
    MAX_LINKED_URLS_TO_FOLLOW, MIN_DEEP_SCRAPE_CONTENT_LENGTH, MIN_DEEP_SCRAPE_RELEVANCE_SCORE,
};
#[cfg(feature = "net")]
use crate::net::{Browser, user_agent_by_seed};
#[cfg(feature = "net")]
use crate::tram::scraper::scrape_and_ingest_with_redirect;
#[cfg(feature = "net")]
use tokio::time::{Duration, sleep};
#[cfg(feature = "net")]
use tracing::info;

/// Result of a deep scraping operation.
#[derive(Debug, Clone, Default)]
pub struct DeepScrapeResult {
    /// Total number of URLs where scraping was attempted.
    pub urls_attempted: usize,
    /// Number of URLs that yielded topic-relevant content above the threshold.
    pub relevant_content_found: usize,
    /// Total knowledge-index chunks ingested across all URLs.
    pub total_chunks_ingested: usize,
    /// Number of scraping iterations completed.
    pub iterations_performed: usize,
    /// URLs from which content was successfully extracted.
    pub successful_urls: Vec<String>,
    /// URLs that were attempted but yielded no extractable content.
    pub unscraped_urls: Vec<String>,
    /// Every URL that was visited, in order of first encounter.
    pub all_scraped_urls: Vec<String>,
}

/// Calculates topic relevance score based on keyword matching.
///
/// Returns a value in `[0.0, 1.0]` combining keyword density (40 %),
/// vocabulary coverage (40 %) and content-length adequacy (20 %).
///
/// # Arguments
///
/// * `content` - The scraped content to evaluate.
/// * `topic`   - Space-separated topic keywords to match against.
#[cfg(feature = "net")]
fn calculate_topic_relevance(content: &str, topic: &str) -> f64 {
    let content_lower = content.to_ascii_lowercase();
    let topic_lower = topic.to_ascii_lowercase();

    let topic_keywords: Vec<&str> = topic_lower
        .split_whitespace()
        .filter(|w| w.len() >= DEEP_SCRAPE_RELEVANCE_KEYWORD_MIN_LENGTH)
        .collect();

    if topic_keywords.is_empty() {
        return 0.0;
    }

    let word_count = content.split_whitespace().count();
    if word_count == 0 {
        return 0.0;
    }

    let keyword_matches: usize = topic_keywords
        .iter()
        .map(|k| content_lower.matches(k).count())
        .sum();

    let keyword_density = (keyword_matches as f64) / (word_count as f64) * 100.0;

    let unique_matches = topic_keywords
        .iter()
        .filter(|&&k| content_lower.contains(k))
        .count();
    let coverage = (unique_matches as f64) / (topic_keywords.len() as f64);

    let length_factor = if word_count >= MIN_DEEP_SCRAPE_CONTENT_LENGTH {
        1.0
    } else {
        word_count as f64 / MIN_DEEP_SCRAPE_CONTENT_LENGTH as f64
    };

    (keyword_density / 10.0).min(1.0) * 0.4 + coverage * 0.4 + length_factor * 0.2
}

/// Decodes a DuckDuckGo Lite redirect URL into the actual target URL.
///
/// DDG Lite returns hrefs in the form `/l/?uddg=https%3A%2F%2Fexample.com%2F&rut=...`.
/// This function extracts and percent-decodes the `uddg` parameter value.
/// If the input is already a plain URL (starts with `http`), it is returned as-is.
///
/// # Arguments
///
/// * `href` - The raw href string from a DDG Lite search result anchor tag.
///
/// # Returns
///
/// The resolved absolute URL string, or the original `href` unchanged when no
/// redirect encoding is detected.
#[cfg(feature = "net")]
fn decode_ddg_redirect(href: &str) -> String {
    if let Some(pos) = href.find(DEEP_SCRAPE_DDG_REDIRECT_PARAM) {
        let encoded_url = &href[pos + DEEP_SCRAPE_DDG_REDIRECT_PARAM.len()..];
        let end = encoded_url.find('&').unwrap_or(encoded_url.len());
        let encoded_url = &encoded_url[..end];
        percent_decode(encoded_url)
    } else {
        href.to_string()
    }
}

/// Percent-decodes a URL-encoded string (e.g. `%2F` → `/`, `%3A` → `:`).
///
/// Only handles the subset needed for DDG redirect URL decoding.
#[cfg(feature = "net")]
fn percent_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2]))
        {
            result.push(char::from(h << 4 | l));
            i += 3;
            continue;
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Returns the nibble value of an ASCII hex digit, or `None` if not a hex digit.
#[cfg(feature = "net")]
fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Collects candidate URLs for a single iteration using the multi-tier strategy.
///
/// Tier 1 (DDG Lite) is always attempted first; Tiers 2 and 3 (DDG Instant
/// Answer API) are appended as supplements.
///
/// # Arguments
///
/// * `browser`       - Configured HTTP client.
/// * `query`         - The search query for this iteration.
/// * `encoded`       - URL-encoded form of `query` (spaces → `+`).
/// * `ua`            - User-agent string to use for this iteration.
/// * `scraped_urls`  - Set of URLs already visited; candidates in this set are skipped.
#[cfg(feature = "net")]
async fn collect_urls_for_iteration(
    browser: &Browser,
    query: &str,
    encoded: &str,
    ua: &str,
    scraped_urls: &std::collections::HashSet<String>,
) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();

    if let Ok(lite_results) = browser
        .lite_search(
            query,
            DEEP_SCRAPE_LITE_QUERY_REGION,
            Some(MAX_DEEP_SCRAPE_LITE_RESULTS),
            ua,
        )
        .await
    {
        for r in &lite_results {
            let resolved = decode_ddg_redirect(&r.url);
            if resolved.starts_with(DEEP_SCRAPE_HTTP_PREFIX)
                && !resolved.contains(DEEP_SCRAPE_DOMAIN_FILTER)
                && !scraped_urls.contains(&resolved)
                && !urls.contains(&resolved)
            {
                urls.push(resolved);
            }
        }
    }

    let ia_path = format!("?q={}&ia={}", encoded, DEEP_SCRAPE_IA_PARAM_WEB);
    if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
        if let Some(abstract_url) = resp.abstract_url.as_ref()
            && abstract_url.starts_with(DEEP_SCRAPE_HTTP_PREFIX)
            && !abstract_url.contains(DEEP_SCRAPE_DOMAIN_FILTER)
            && !scraped_urls.contains(abstract_url)
            && !urls.contains(abstract_url)
        {
            urls.push(abstract_url.clone());
        }

        if let Some(definition_url) = resp.definition_url.as_ref()
            && definition_url.starts_with(DEEP_SCRAPE_HTTP_PREFIX)
            && !definition_url.contains(DEEP_SCRAPE_DOMAIN_FILTER)
            && !scraped_urls.contains(definition_url)
            && !urls.contains(definition_url)
        {
            urls.push(definition_url.clone());
        }

        for topic_entry in resp
            .related_topics
            .iter()
            .take(DEEP_SCRAPE_MAX_RELATED_TOPICS)
        {
            if let Some(url) = topic_entry.first_url.as_ref()
                && url.starts_with(DEEP_SCRAPE_HTTP_PREFIX)
                && !url.contains(DEEP_SCRAPE_DOMAIN_FILTER)
                && !scraped_urls.contains(url)
                && !urls.contains(url)
            {
                urls.push(url.clone());
            }
        }
    }

    urls
}

/// Builds the ordered list of search query strings for a topic.
///
/// Returns more varied queries than the previous implementation to increase
/// the chance of finding topic-relevant pages via DDG Lite.
///
/// # Arguments
///
/// * `topic` - The topic keyword string.
#[cfg(feature = "net")]
fn build_query_prefixes(topic: &str) -> Vec<String> {
    vec![
        format!("advantages of {}", topic),
        format!("benefits of {}", topic),
        format!("pros and cons of {}", topic),
        format!("{} tutorial", topic),
        format!("{} guide", topic),
        format!("{} overview", topic),
    ]
}

/// Performs deep scraping for topic-relevant content using a multi-tier URL strategy.
///
/// Each iteration issues a DDG Lite search (Tier 1) plus a DDG Instant Answer
/// API call (Tiers 2-3) to collect candidate page URLs. Each URL is then
/// scraped and its content injected into `index`. An exponential back-off delay
/// is applied between iterations to respect rate limits.
///
/// # Arguments
///
/// * `topic`          - The topic to search for (e.g., `"rust programming language"`).
/// * `index`          - The knowledge index to ingest content into.
/// * `verbose`        - When `true`, emits detailed progress via `tracing::info`.
/// * `max_iterations` - Maximum number of search-and-scrape cycles to run.
///
/// # Returns
///
/// A [`DeepScrapeResult`] containing ingestion statistics and URL lists.
#[cfg(feature = "net")]
pub async fn deep_scrape_for_topic(
    topic: &str,
    index: &mut KnowledgeIndex,
    verbose: bool,
    max_iterations: usize,
) -> DeepScrapeResult {
    let mut result = DeepScrapeResult::default();
    let browser = Browser::new();
    let mut scraped_urls: std::collections::HashSet<String> = std::collections::HashSet::new();

    let topic_keywords: Vec<String> = topic
        .split_whitespace()
        .filter(|w| w.len() >= DEEP_SCRAPE_KEYWORD_MIN_LENGTH)
        .map(|w| w.to_string())
        .collect();

    if verbose {
        info!(
            "  🔍 Deep Search: topic keywords extracted: {:?}",
            topic_keywords
        );
    }

    if topic_keywords.is_empty() {
        if verbose {
            info!(
                "  🔍 Deep Search: no valid topic keywords found (topic='{}')",
                topic
            );
        }
        return result;
    }

    let query_prefixes = build_query_prefixes(topic);
    let iterations = max_iterations.min(MAX_DEEP_SCRAPE_ITERATIONS);
    let mut all_scraped_urls: Vec<String> = Vec::new();

    if verbose {
        info!(
            "  🔍 Deep Search: topic='{}', max_iterations={}",
            topic, iterations
        );
    }

    for iteration in 0..iterations {
        let query = query_prefixes[iteration % query_prefixes.len()].clone();
        let encoded: String = query
            .chars()
            .map(|c| if c == ' ' { '+' } else { c })
            .collect();

        let ua = user_agent_by_seed(iteration as u64);

        if verbose {
            info!(
                "  🔍 Deep Search iteration {}: query='{}'",
                iteration + 1,
                query
            );
        }

        let urls_to_scrape =
            collect_urls_for_iteration(&browser, &query, &encoded, ua, &scraped_urls).await;

        result.iterations_performed += 1;

        let urls_this_iteration: Vec<String> = urls_to_scrape
            .into_iter()
            .take(DEEP_SCRAPE_MAX_URLS_PER_ITERATION)
            .collect();

        if verbose {
            if urls_this_iteration.is_empty() {
                info!("  🔍 Deep Search: query='{}', found 0 URLs", query);
            } else {
                info!(
                    "  🔍 Deep Search: query='{}', found {} URL(s)",
                    query,
                    urls_this_iteration.len()
                );
            }
        }

        for url in &urls_this_iteration {
            if scraped_urls.contains(url) {
                continue;
            }

            result.urls_attempted += 1;
            scraped_urls.insert(url.clone());
            all_scraped_urls.push(url.clone());

            let (scraped, linked_urls) =
                scrape_and_ingest_with_redirect(url, index, DEEP_SCRAPE_SOURCE_LABEL, topic).await;

            if scraped > 0 {
                result.total_chunks_ingested += scraped;
                result.successful_urls.push(url.clone());

                let kb_corpus: String = index
                    .query(topic, DEEP_SCRAPE_KB_QUERY_LIMIT)
                    .into_iter()
                    .map(|c| c.text.clone())
                    .collect::<Vec<_>>()
                    .join(" ");

                if calculate_topic_relevance(&kb_corpus, topic) >= MIN_DEEP_SCRAPE_RELEVANCE_SCORE {
                    result.relevant_content_found += 1;
                }

                for linked_url in linked_urls.iter().take(MAX_LINKED_URLS_TO_FOLLOW) {
                    if !scraped_urls.contains(linked_url) {
                        scraped_urls.insert(linked_url.clone());
                        all_scraped_urls.push(linked_url.clone());
                        let (linked_scraped, _) = scrape_and_ingest_with_redirect(
                            linked_url,
                            index,
                            DEEP_SCRAPE_SOURCE_LABEL,
                            topic,
                        )
                        .await;
                        if linked_scraped > 0 {
                            result.total_chunks_ingested += linked_scraped;
                        } else {
                            result.unscraped_urls.push(linked_url.clone());
                        }
                    }
                }
            } else {
                result.unscraped_urls.push(url.clone());
            }
        }

        if iteration + 1 < iterations {
            sleep(Duration::from_millis(
                DEEP_SCRAPE_BACK_OFF_BASE_MS * (iteration as u64 + 1),
            ))
            .await;
        }
    }

    if verbose {
        info!(
            "  🔍 Deep Search: completed iterations={}, urls_attempted={}, relevant_found={}",
            result.iterations_performed, result.urls_attempted, result.relevant_content_found
        );
        if !result.successful_urls.is_empty() {
            info!(
                "  🔍 Deep Search: scraped URLs: {:?}",
                result.successful_urls
            );
        }
    }

    result.all_scraped_urls = all_scraped_urls;
    result
}

#[cfg(not(feature = "net"))]
pub async fn deep_scrape_for_topic(
    _topic: &str,
    _index: &mut KnowledgeIndex,
    _verbose: bool,
    _max_iterations: usize,
) -> DeepScrapeResult {
    DeepScrapeResult::default()
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
