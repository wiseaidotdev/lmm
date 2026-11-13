// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Deep Web Scraping for Topic-Relevant Content
//!
//! Iteratively scrapes websites from DuckDuckGo `!ducky` results to find
//! content similar to the topic being asked, and injects it into the knowledge corpus.

use crate::cognition::knowledge::KnowledgeIndex;
#[cfg(feature = "net")]
use crate::constants::tram_web_fetch::{
    DEEP_SCRAPE_DOMAIN_FILTER, DEEP_SCRAPE_HTTP_PREFIX, DEEP_SCRAPE_IA_PARAM_WEB,
    DEEP_SCRAPE_KB_QUERY_LIMIT, DEEP_SCRAPE_KEYWORD_MIN_LENGTH, DEEP_SCRAPE_MAX_RELATED_TOPICS,
    DEEP_SCRAPE_MAX_URLS_PER_ITERATION, DEEP_SCRAPE_RELEVANCE_KEYWORD_MIN_LENGTH,
    DEEP_SCRAPE_SOURCE_LABEL, MAX_DEEP_SCRAPE_ITERATIONS, MAX_LINKED_URLS_TO_FOLLOW,
    MIN_DEEP_SCRAPE_CONTENT_LENGTH, MIN_DEEP_SCRAPE_RELEVANCE_SCORE,
};
#[cfg(feature = "net")]
use crate::net::Browser;
#[cfg(feature = "net")]
use crate::tram::scraper::scrape_and_ingest_with_redirect;
#[cfg(feature = "net")]
use tracing::info;

/// Result of a deep scraping operation.
#[derive(Debug, Clone, Default)]
pub struct DeepScrapeResult {
    pub urls_attempted: usize,
    pub relevant_content_found: usize,
    pub total_chunks_ingested: usize,
    pub iterations_performed: usize,
    pub successful_urls: Vec<String>,
    pub unscraped_urls: Vec<String>,
    pub all_scraped_urls: Vec<String>,
}

/// Calculates topic relevance score based on keyword matching.
///
/// # Arguments
///
/// * `content` - The scraped content to evaluate.
/// * `topic` - The topic keywords to match against.
///
/// # Returns
///
/// A relevance score between 0.0 and 1.0.
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

    let mut keyword_matches = 0usize;
    for keyword in &topic_keywords {
        let count = content_lower.matches(keyword).count();
        keyword_matches += count;
    }

    let keyword_density = (keyword_matches as f64) / (word_count as f64) * 100.0;

    let unique_matches = topic_keywords
        .iter()
        .filter(|&k| content_lower.contains(k))
        .count();
    let coverage = (unique_matches as f64) / (topic_keywords.len() as f64);

    let length_factor = if word_count >= MIN_DEEP_SCRAPE_CONTENT_LENGTH {
        1.0
    } else {
        word_count as f64 / MIN_DEEP_SCRAPE_CONTENT_LENGTH as f64
    };

    (keyword_density / 10.0).min(1.0) * 0.4 + coverage * 0.4 + length_factor * 0.2
}

/// Performs deep scraping for topic-relevant content.
///
/// # Arguments
///
/// * `topic` - The topic to search for.
/// * `index` - The knowledge index to ingest content into.
/// * `verbose` - Whether to show debug output.
/// * `max_iterations` - Maximum number of scraping iterations.
///
/// # Returns
///
/// A `DeepScrapeResult` containing scraping statistics.
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

    let query_prefixes = [
        format!("advantages of {}", topic),
        format!("benefits of {}", topic),
        format!("pros and cons of {}", topic),
    ];

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
        let ia_path = format!("?q={}&ia={}", encoded, DEEP_SCRAPE_IA_PARAM_WEB);

        if verbose {
            info!(
                "  🔍 Deep Search iteration {}: query='{}'",
                iteration + 1,
                encoded
            );
        }

        match browser.get_api_response(&ia_path, None).await {
            Ok(resp) => {
                result.iterations_performed += 1;

                let mut urls_to_scrape: Vec<String> = Vec::new();

                if let Some(abstract_url) = resp.abstract_url.as_ref()
                    && abstract_url.starts_with(DEEP_SCRAPE_HTTP_PREFIX)
                    && !abstract_url.contains(DEEP_SCRAPE_DOMAIN_FILTER)
                {
                    urls_to_scrape.push(abstract_url.clone());
                }

                if let Some(definition_url) = resp.definition_url.as_ref()
                    && definition_url.starts_with(DEEP_SCRAPE_HTTP_PREFIX)
                    && !definition_url.contains(DEEP_SCRAPE_DOMAIN_FILTER)
                {
                    urls_to_scrape.push(definition_url.clone());
                }

                for topic_entry in resp
                    .related_topics
                    .iter()
                    .take(DEEP_SCRAPE_MAX_RELATED_TOPICS)
                {
                    if let Some(url) = topic_entry.first_url.as_ref()
                        && url.starts_with(DEEP_SCRAPE_HTTP_PREFIX)
                        && !url.contains(DEEP_SCRAPE_DOMAIN_FILTER)
                    {
                        urls_to_scrape.push(url.clone());
                    }
                }

                let urls_attempted_this_iteration: Vec<String> = urls_to_scrape
                    .iter()
                    .take(DEEP_SCRAPE_MAX_URLS_PER_ITERATION)
                    .cloned()
                    .collect();
                let mut successful_urls_this_iteration = Vec::new();
                let mut unscraped_urls_this_iteration = Vec::new();

                for url in &urls_attempted_this_iteration {
                    if scraped_urls.contains(url) {
                        continue;
                    }

                    result.urls_attempted += 1;
                    scraped_urls.insert(url.clone());
                    all_scraped_urls.push(url.clone());

                    let (scraped, linked_urls) = scrape_and_ingest_with_redirect(
                        url,
                        index,
                        DEEP_SCRAPE_SOURCE_LABEL,
                        topic,
                    )
                    .await;

                    if scraped > 0 {
                        result.total_chunks_ingested += scraped;
                        result.successful_urls.push(url.clone());
                        successful_urls_this_iteration.push(url.clone());

                        let kb_corpus: String = index
                            .query(topic, DEEP_SCRAPE_KB_QUERY_LIMIT)
                            .into_iter()
                            .map(|c| c.text.clone())
                            .collect::<Vec<_>>()
                            .join(" ");

                        let relevance = calculate_topic_relevance(&kb_corpus, topic);

                        if relevance >= MIN_DEEP_SCRAPE_RELEVANCE_SCORE {
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
                        unscraped_urls_this_iteration.push(url.clone());
                    }
                }

                if verbose {
                    if urls_attempted_this_iteration.is_empty() {
                        info!("  🔍 Deep Search: tried query='{}', found 0 URLs", encoded);
                    } else {
                        let urls_str: Vec<&str> = urls_attempted_this_iteration
                            .iter()
                            .map(|u| u.as_str())
                            .collect();
                        info!(
                            "  🔍 Deep Search: tried query='{}', found {} URLs: {}",
                            encoded,
                            urls_str.len(),
                            urls_str.join(", ")
                        );
                    }
                }
            }
            Err(e) => {
                if verbose {
                    info!("  🔍 Deep Search: Instant Answer API failed: {}", e);
                }
            }
        }
    }

    if verbose {
        info!(
            "  🔍 Deep Search: completed iterations={}, urls_attempted={}, relevant_found={}",
            result.iterations_performed, result.urls_attempted, result.relevant_content_found
        );
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
