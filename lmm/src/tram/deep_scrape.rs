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

#[cfg(any(
    all(feature = "net", not(target_arch = "wasm32")),
    not(feature = "net")
))]
use crate::cognition::knowledge::KnowledgeIndex;
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use crate::cognition::task_decomposer::{CreativeOutputKind, TaskDecomposer};
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use crate::cognition::thought::classifier::{InputClassifier, ReasoningMode};
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use crate::constants::tram_web_fetch::{
    DEEP_QUERY_ADVANTAGES, DEEP_QUERY_BENEFITS, DEEP_QUERY_CAUSE, DEEP_QUERY_CODE_IMPL,
    DEEP_QUERY_CODE_SNIPPET, DEEP_QUERY_COMPARE, DEEP_QUERY_COMPUTE_IN, DEEP_QUERY_DEFINITION,
    DEEP_QUERY_DIFFERENCES, DEEP_QUERY_EXAMPLE, DEEP_QUERY_EXPERIENCE, DEEP_QUERY_EXPLAINED,
    DEEP_QUERY_FEATURES, DEEP_QUERY_FUNCTION, DEEP_QUERY_GUIDE, DEEP_QUERY_HOW_TO,
    DEEP_QUERY_HOW_TO_IMPLEMENT, DEEP_QUERY_MEANING, DEEP_QUERY_METAPHORS, DEEP_QUERY_OVERVIEW,
    DEEP_QUERY_PHILOSOPHY, DEEP_QUERY_PROGRAM, DEEP_QUERY_PROS_CONS, DEEP_QUERY_REASON,
    DEEP_QUERY_RECURSIVE, DEEP_QUERY_THEMES, DEEP_QUERY_TUTORIAL, DEEP_QUERY_VS,
    DEEP_QUERY_WHAT_IS, DEEP_QUERY_WHY, DEEP_QUERY_WHY_USE, DEEP_SCRAPE_ABS_REDIRECT_PREFIX,
    DEEP_SCRAPE_BACK_OFF_BASE_MS, DEEP_SCRAPE_DDG_BASE, DEEP_SCRAPE_DDG_ENTITY_PREFIX,
    DEEP_SCRAPE_DDG_REDIRECT_PARAM, DEEP_SCRAPE_DOMAIN_FILTER, DEEP_SCRAPE_HTTP_PREFIX,
    DEEP_SCRAPE_IA_PARAM_WEB, DEEP_SCRAPE_KB_QUERY_LIMIT, DEEP_SCRAPE_KEYWORD_MIN_LENGTH,
    DEEP_SCRAPE_LITE_QUERY_REGION, DEEP_SCRAPE_MAX_RELATED_TOPICS, DEEP_SCRAPE_MAX_RETRIES,
    DEEP_SCRAPE_MAX_URLS_PER_ITERATION, DEEP_SCRAPE_NO_RESULTS_MSG, DEEP_SCRAPE_PROTOCOL_HTTPS,
    DEEP_SCRAPE_PROTOCOL_RELATIVE_PREFIX, DEEP_SCRAPE_RELEVANCE_KEYWORD_MIN_LENGTH,
    DEEP_SCRAPE_RETRY_BASE_MS, DEEP_SCRAPE_SOURCE_LABEL, DEEP_SCRAPE_WIKIPEDIA_PREFIX,
    MAX_DEEP_SCRAPE_ITERATIONS, MAX_DEEP_SCRAPE_LITE_RESULTS, MAX_LINKED_URLS_TO_FOLLOW,
    MIN_DEEP_SCRAPE_CONTENT_LENGTH, MIN_DEEP_SCRAPE_RELEVANCE_SCORE,
    OPENSEARCH_QUERY_ALGORITHM_LANG_TEMPLATE, OPENSEARCH_QUERY_IMPLEMENTATION_TEMPLATE,
    OPENSEARCH_QUERY_PROGRAM_LANG_TEMPLATE, OPENSEARCH_QUERY_PROGRAM_TEMPLATE,
    WIKIPEDIA_FALLBACK_ALGORITHM_SUFFIX, WIKIPEDIA_FALLBACK_COMPUTATION_SUFFIX,
    WIKIPEDIA_FALLBACK_EXAMPLE_SUFFIX, WIKIPEDIA_FALLBACK_FUNCTION_SUFFIX,
    WIKIPEDIA_FALLBACK_METHOD_SUFFIX, WIKIPEDIA_FALLBACK_NUMERICAL_PREFIX,
    WIKIPEDIA_FALLBACK_PROGRAM_SUFFIX, WIKIPEDIA_OPENSEARCH_API,
};
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use crate::net::{Browser, user_agent_by_seed};
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use crate::tram::query::extract_creative_subject;
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use crate::tram::scraper::scrape_and_ingest_with_redirect;
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
use tokio::time::{Duration, sleep, timeout};
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
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
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
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

/// Resolves a raw DDG Lite href into an absolute HTTP(S) URL.
///
/// Handles four href formats returned by DDG Lite:
///
/// | Format                     | Example                                        | Resolution                     |
/// |----------------------------|-------------------------------------------------|-------------------------------|
/// | Protocol-relative redirect | `//duckduckgo.com/l/?uddg=https%3A...`             | Prepend `https:`, decode `uddg` |
/// | Absolute redirect path     | `/l/?uddg=https%3A...`                            | Prepend DDG base, decode `uddg` |
/// | Direct HTTP URL            | `https://example.com/article`                   | Pass through                   |
/// | DDG entity page            | `https://duckduckgo.com/Rust_(programming_language)` | Map to Wikipedia URL     |
///
/// Returns an empty string when no usable URL can be extracted.
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
pub fn resolve_ddg_url(href: &str) -> String {
    let normalised = if href.starts_with(DEEP_SCRAPE_PROTOCOL_RELATIVE_PREFIX) {
        format!("{}{}", DEEP_SCRAPE_PROTOCOL_HTTPS, href)
    } else if href.starts_with(DEEP_SCRAPE_ABS_REDIRECT_PREFIX) {
        format!("{}{}", DEEP_SCRAPE_DDG_BASE, href)
    } else {
        href.to_string()
    };

    if let Some(pos) = normalised.find(DEEP_SCRAPE_DDG_REDIRECT_PARAM) {
        let encoded_url = &normalised[pos + DEEP_SCRAPE_DDG_REDIRECT_PARAM.len()..];
        let end = encoded_url.find('&').unwrap_or(encoded_url.len());
        let decoded = percent_decode(&encoded_url[..end]);
        if decoded.starts_with(DEEP_SCRAPE_HTTP_PREFIX) {
            return decoded;
        }
    }

    if let Some(path) = normalised.strip_prefix(DEEP_SCRAPE_DDG_ENTITY_PREFIX)
        && !path.is_empty()
        && !path.starts_with("l/")
        && !path.starts_with("?")
        && !path.starts_with("lite")
        && !path.contains("duckduckgo")
    {
        return format!("{}{}", DEEP_SCRAPE_WIKIPEDIA_PREFIX, path);
    }

    if normalised.starts_with(DEEP_SCRAPE_HTTP_PREFIX)
        && !normalised.contains(DEEP_SCRAPE_DOMAIN_FILTER)
    {
        return normalised;
    }

    String::new()
}

/// Percent-decodes a URL-encoded string (e.g. `%2F` → `/`, `%3A` → `:`).
///
/// Only handles the subset needed for DDG redirect URL decoding.
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
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
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
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
/// Tier 1 (DDG Lite) is always attempted first with exponential back-off
/// retries on failure; Tiers 2 and 3 (DDG Instant Answer API) are appended
/// as supplements.
///
/// # Arguments
///
/// * `browser`       - Configured HTTP client.
/// * `query`         - The search query for this iteration.
/// * `encoded`       - URL-encoded form of `query` (spaces → `+`).
/// * `ua_seed`       - Seed for generating user-agent strings (incremented on retry).
/// * `scraped_urls`  - Set of URLs already visited; candidates in this set are skipped.
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
async fn collect_urls_for_iteration(
    browser: &Browser,
    query: &str,
    encoded: &str,
    ua_seed: u64,
    scraped_urls: &std::collections::HashSet<String>,
) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();

    let mut attempt: u32 = 0;
    loop {
        let ua = user_agent_by_seed(ua_seed + attempt as u64);
        match browser
            .lite_search(
                query,
                DEEP_SCRAPE_LITE_QUERY_REGION,
                Some(MAX_DEEP_SCRAPE_LITE_RESULTS),
                ua,
            )
            .await
        {
            Ok(lite_results) => {
                info!(
                    "  🔍 Deep Search: lite_search returned {} href(s) for '{}'",
                    lite_results.len(),
                    query
                );
                if lite_results.is_empty() {
                    info!("  🔍 Deep Search: {}", DEEP_SCRAPE_NO_RESULTS_MSG);
                }
                for r in &lite_results {
                    info!("  🔍 Deep Search raw href: {:?}", r.url);
                    let resolved = resolve_ddg_url(&r.url);
                    if !resolved.is_empty()
                        && !resolved.contains(DEEP_SCRAPE_DOMAIN_FILTER)
                        && !scraped_urls.contains(&resolved)
                        && !urls.contains(&resolved)
                    {
                        urls.push(resolved);
                    }
                }
                break;
            }
            Err(_) if attempt < DEEP_SCRAPE_MAX_RETRIES => {
                attempt += 1;
                sleep(Duration::from_millis(
                    DEEP_SCRAPE_RETRY_BASE_MS * 2u64.pow(attempt),
                ))
                .await;
            }
            Err(_) => break,
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
/// Returns mode-aware queries adapted to the prompt's reasoning mode,
/// increasing the chance of finding relevant pages via DDG Lite.
///
/// # Arguments
///
/// * `topic`  - The topic keyword string.
/// * `prompt` - The original user prompt, used for mode classification.
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
fn build_query_prefixes(topic: &str, prompt: &str) -> Vec<String> {
    let mode = InputClassifier::classify(prompt);
    match mode {
        ReasoningMode::Evaluative => vec![
            format!("{}{}", DEEP_QUERY_ADVANTAGES, topic),
            format!("{}{}", DEEP_QUERY_BENEFITS, topic),
            format!("{}{}", topic, DEEP_QUERY_FEATURES),
            format!("{}{}", DEEP_QUERY_WHY_USE, topic),
            format!("{}{}", topic, DEEP_QUERY_PROS_CONS),
            format!("{}{}", topic, DEEP_QUERY_OVERVIEW),
        ],
        ReasoningMode::Causal => vec![
            format!("{}{}", DEEP_QUERY_WHY, topic),
            format!("{}{}", topic, DEEP_QUERY_CAUSE),
            format!("{}{}", topic, DEEP_QUERY_REASON),
            format!("{}{}", DEEP_QUERY_WHAT_IS, topic),
            format!("{}{}", topic, DEEP_QUERY_EXPLAINED),
            format!("{}{}", topic, DEEP_QUERY_OVERVIEW),
        ],
        ReasoningMode::Procedural => vec![
            format!("{}{}", DEEP_QUERY_HOW_TO, topic),
            format!("{}{}", topic, DEEP_QUERY_TUTORIAL),
            format!("{}{}", topic, DEEP_QUERY_GUIDE),
            format!("{}{}", DEEP_QUERY_WHAT_IS, topic),
            format!("{}{}", topic, DEEP_QUERY_OVERVIEW),
            topic.to_string(),
        ],
        ReasoningMode::Analogical => vec![
            format!("{}{}", DEEP_QUERY_COMPARE, topic),
            format!("{}{}", DEEP_QUERY_DIFFERENCES, topic),
            format!("{}{}{}", topic, DEEP_QUERY_VS, "alternatives"),
            format!("{}{}", topic, DEEP_QUERY_FEATURES),
            format!("{}{}", topic, DEEP_QUERY_OVERVIEW),
            topic.to_string(),
        ],
        ReasoningMode::Definitional | ReasoningMode::FactualLookup => vec![
            format!("{}{}", DEEP_QUERY_WHAT_IS, topic),
            format!("{}{}", topic, DEEP_QUERY_DEFINITION),
            format!("{}{}", topic, DEEP_QUERY_EXPLAINED),
            format!("{}{}", topic, DEEP_QUERY_OVERVIEW),
            format!("{}{}", topic, DEEP_QUERY_GUIDE),
            topic.to_string(),
        ],
        ReasoningMode::TaskExecution => {
            let subject = extract_creative_subject(prompt);
            let kind = TaskDecomposer::detect_creative_kind(prompt);
            match kind {
                CreativeOutputKind::Code { ref language } => vec![
                    format!("{} {}{}", language, subject, DEEP_QUERY_PROGRAM),
                    format!("{} {}{}", language, subject, DEEP_QUERY_FUNCTION),
                    format!("{}{} in {}", DEEP_QUERY_HOW_TO_IMPLEMENT, subject, language),
                    format!("{} {} {}", DEEP_QUERY_COMPUTE_IN, subject, language),
                    format!("{} {}{}", language, subject, DEEP_QUERY_CODE_IMPL),
                    format!("{} {}{}", language, subject, DEEP_QUERY_RECURSIVE),
                    format!("{} {}{}", language, subject, DEEP_QUERY_EXAMPLE),
                    format!("{} {}{}", language, subject, DEEP_QUERY_CODE_SNIPPET),
                    format!("{} {}{}", language, subject, DEEP_QUERY_TUTORIAL),
                    format!("{} {}", subject, language),
                ],
                CreativeOutputKind::Poem => vec![
                    format!("{}{}", subject, DEEP_QUERY_THEMES),
                    format!("{}{}", subject, DEEP_QUERY_MEANING),
                    format!("{}{}", subject, DEEP_QUERY_PHILOSOPHY),
                    format!("{}{}", subject, DEEP_QUERY_METAPHORS),
                    format!("{}{}", subject, DEEP_QUERY_EXPERIENCE),
                    subject.to_string(),
                ],
                _ => vec![
                    format!("{}{}", subject, DEEP_QUERY_OVERVIEW),
                    format!("{}{}", subject, DEEP_QUERY_EXPLAINED),
                    format!("{}{}", subject, DEEP_QUERY_MEANING),
                    subject.to_string(),
                ],
            }
        }
        _ => vec![
            format!("{}{}", DEEP_QUERY_WHAT_IS, topic),
            format!("{}{}", topic, DEEP_QUERY_OVERVIEW),
            format!("{}{}", topic, DEEP_QUERY_EXPLAINED),
            format!("{}{}", topic, DEEP_QUERY_GUIDE),
            format!("{}{}", topic, DEEP_QUERY_TUTORIAL),
            topic.to_string(),
        ],
    }
}

/// Queries the Wikipedia OpenSearch API and returns article URLs for a given query.
///
/// Uses Wikipedia's free `opensearch` API endpoint which returns structured JSON
/// containing matched article titles and their canonical URLs.  This avoids the
/// need to guess URL patterns (e.g., `Numerical_differentiation` for `derivative`).
///
/// Returns up to 5 article URLs sorted by search relevance, or an empty vector
/// when the request fails or returns no results.
///
/// # Arguments
///
/// * `query` - The search query (e.g., `"numerical derivative algorithm"`).
///
/// # Time Complexity
///
/// O(1) locally; network-bound.
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
async fn fetch_wikipedia_opensearch_urls(query: &str) -> Vec<String> {
    let seed: u64 = query
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let client = match reqwest::Client::builder()
        .user_agent(user_agent_by_seed(seed))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let encoded: String = query
        .chars()
        .flat_map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                vec![c]
            } else if c == ' ' {
                vec!['+']
            } else {
                format!("%{:02X}", c as u32).chars().collect::<Vec<char>>()
            }
        })
        .collect();

    let url = format!("{}{}", WIKIPEDIA_OPENSEARCH_API, encoded);

    let body = match timeout(Duration::from_secs(5), client.get(&url).send()).await {
        Ok(Ok(r)) => match r.text().await {
            Ok(t) => t,
            Err(_) => return vec![],
        },
        _ => return vec![],
    };

    extract_wikipedia_urls_from_opensearch(&body)
}

/// Parses a Wikipedia OpenSearch JSON response and extracts article URLs.
///
/// The OpenSearch format is `["query", ["title1",...], ["desc1",...], ["url1",...]]`.
/// Only URLs pointing to `en.wikipedia.org/wiki/` are returned.
///
/// # Time Complexity
///
/// O(|body|)
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
fn extract_wikipedia_urls_from_opensearch(body: &str) -> Vec<String> {
    let trimmed = body.trim();
    let inner = match trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        Some(s) => s,
        None => return vec![],
    };

    let url_prefix = "\"https://en.wikipedia.org/wiki/";
    let mut urls = Vec::new();
    let mut pos = 0;
    while let Some(start) = inner[pos..].find(url_prefix) {
        let abs_start = pos + start + 1;
        let after_quote = abs_start + "https://en.wikipedia.org/wiki/".len();
        if let Some(end_rel) = inner[after_quote..].find('"') {
            let url = &inner[abs_start..after_quote + end_rel];
            urls.push(url.to_string());
            pos = after_quote + end_rel + 1;
        } else {
            break;
        }
    }
    urls
}

/// Builds a list of Wikipedia article URLs to scrape as a guaranteed fallback.
///
/// For task-execution (code-writing) prompts, constructs Wikipedia URLs from
/// the subject extracted from the prompt.  The subject is title-cased to match
/// Wikipedia's URL convention.  Multiple URL forms are generated to increase the
/// chance of finding a matching article:
/// - `{WIKIPEDIA_PREFIX}{Subject_Title_Case}` (e.g. `Factorial`)
/// - `{WIKIPEDIA_PREFIX}{subject_lowercase}` (e.g. `factorial`)
/// - `{WIKIPEDIA_PREFIX}Numerical_{Subject}` (e.g. `Numerical_Derivative`)
/// - `{WIKIPEDIA_PREFIX}{Subject}_algorithm` etc.
///
/// Wikipedia is a reliable source that can be scraped without CAPTCHA and
/// frequently contains `<pre>` pseudocode blocks, formula descriptions, and
/// algorithmic prose that the code-block extractor can harvest.
///
/// # Time Complexity
///
/// O(|prompt|)
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
fn build_wikipedia_fallback_urls(prompt: &str, topic: &str) -> Vec<String> {
    let kind = TaskDecomposer::detect_creative_kind(prompt);
    let subject = match &kind {
        CreativeOutputKind::Code { .. } => extract_creative_subject(prompt),
        _ => topic.to_string(),
    };

    if subject.is_empty() {
        return vec![];
    }

    let title_case: String = subject
        .split(['_', ' '])
        .filter(|w| !w.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().to_string() + chars.as_str(),
            }
        })
        .collect::<Vec<String>>()
        .join("_");

    let title_spaced: String = subject.replace('_', " ");
    let title_spaced_cap: String = {
        let mut chars = title_spaced.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().to_string() + chars.as_str(),
        }
    };

    let lc = subject.to_ascii_lowercase().replace(' ', "_");
    let spaced = title_spaced_cap.replace(' ', "_");

    let subject_is_single_word = !subject.contains(' ') && !subject.contains('_');

    let mut urls = vec![
        format!("{}{}", DEEP_SCRAPE_WIKIPEDIA_PREFIX, title_case.clone()),
        format!("{}{}", DEEP_SCRAPE_WIKIPEDIA_PREFIX, spaced.clone()),
        format!("{}{}", DEEP_SCRAPE_WIKIPEDIA_PREFIX, lc.clone()),
    ];
    urls.push(format!(
        "{}{}{}",
        DEEP_SCRAPE_WIKIPEDIA_PREFIX, title_case, WIKIPEDIA_FALLBACK_PROGRAM_SUFFIX
    ));
    urls.push(format!(
        "{}{}{}",
        DEEP_SCRAPE_WIKIPEDIA_PREFIX, title_case, WIKIPEDIA_FALLBACK_EXAMPLE_SUFFIX
    ));
    if subject_is_single_word {
        urls.push(format!(
            "{}{}{}",
            DEEP_SCRAPE_WIKIPEDIA_PREFIX, WIKIPEDIA_FALLBACK_NUMERICAL_PREFIX, spaced
        ));
        urls.push(format!(
            "{}{}{}",
            DEEP_SCRAPE_WIKIPEDIA_PREFIX, WIKIPEDIA_FALLBACK_NUMERICAL_PREFIX, title_case
        ));
        urls.push(format!(
            "{}{}{}",
            DEEP_SCRAPE_WIKIPEDIA_PREFIX, title_case, WIKIPEDIA_FALLBACK_ALGORITHM_SUFFIX
        ));
        urls.push(format!(
            "{}{}{}",
            DEEP_SCRAPE_WIKIPEDIA_PREFIX, title_case, WIKIPEDIA_FALLBACK_FUNCTION_SUFFIX
        ));
        urls.push(format!(
            "{}{}{}",
            DEEP_SCRAPE_WIKIPEDIA_PREFIX, title_case, WIKIPEDIA_FALLBACK_METHOD_SUFFIX
        ));
        urls.push(format!(
            "{}{}{}",
            DEEP_SCRAPE_WIKIPEDIA_PREFIX, title_case, WIKIPEDIA_FALLBACK_COMPUTATION_SUFFIX
        ));
    }
    urls.dedup();
    urls
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
/// * `prompt`         - The original user prompt, used for mode-aware query generation.
/// * `index`          - The knowledge index to ingest content into.
/// * `verbose`        - When `true`, emits detailed progress via `tracing::info`.
/// * `max_iterations` - Maximum number of search-and-scrape cycles to run.
///
/// # Returns
///
/// A [`DeepScrapeResult`] containing ingestion statistics and URL lists.
#[cfg(all(feature = "net", not(target_arch = "wasm32")))]
pub async fn deep_scrape_for_topic(
    topic: &str,
    prompt: &str,
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

    let query_prefixes = build_query_prefixes(topic, prompt);
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

        if verbose {
            info!(
                "  🔍 Deep Search iteration {}: query='{}'",
                iteration + 1,
                query
            );
        }

        let urls_to_scrape =
            collect_urls_for_iteration(&browser, &query, &encoded, iteration as u64, &scraped_urls)
                .await;

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

    let wikipedia_fallback_urls = build_wikipedia_fallback_urls(prompt, topic);

    let opensearch_queries: Vec<String> = {
        let kind = TaskDecomposer::detect_creative_kind(prompt);
        match &kind {
            CreativeOutputKind::Code { .. } => {
                let subj = extract_creative_subject(prompt);
                vec![
                    OPENSEARCH_QUERY_PROGRAM_TEMPLATE.replace("{}", &subj),
                    OPENSEARCH_QUERY_PROGRAM_LANG_TEMPLATE.replace("{}", &subj),
                    OPENSEARCH_QUERY_ALGORITHM_LANG_TEMPLATE.replace("{}", &subj),
                    OPENSEARCH_QUERY_IMPLEMENTATION_TEMPLATE.replace("{}", &subj),
                ]
            }
            _ => vec![topic.to_string()],
        }
    };

    let mut opensearch_urls: Vec<String> = Vec::new();
    for q in &opensearch_queries {
        if verbose {
            info!("  🔍 Deep Search: Wikipedia OpenSearch query='{}'", q);
        }
        let found = fetch_wikipedia_opensearch_urls(q).await;
        if verbose {
            info!(
                "  🔍 Deep Search: OpenSearch returned {} URL(s): {:?}",
                found.len(),
                found
            );
        }
        for u in found {
            if !opensearch_urls.contains(&u) {
                opensearch_urls.push(u);
            }
        }
    }

    let all_wiki_urls: Vec<String> = {
        let mut combined = wikipedia_fallback_urls;
        for u in opensearch_urls {
            if !combined.contains(&u) {
                combined.push(u);
            }
        }
        combined
    };

    for wiki_url in &all_wiki_urls {
        if scraped_urls.contains(wiki_url) {
            continue;
        }
        scraped_urls.insert(wiki_url.clone());
        all_scraped_urls.push(wiki_url.clone());
        let (scraped, _) =
            scrape_and_ingest_with_redirect(wiki_url, index, DEEP_SCRAPE_SOURCE_LABEL, topic).await;
        if scraped > 0 {
            result.total_chunks_ingested += scraped;
            result.successful_urls.push(wiki_url.clone());
            result.urls_attempted += 1;
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
    _prompt: &str,
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
