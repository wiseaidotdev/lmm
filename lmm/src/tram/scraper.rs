// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Web Scraper
//!
//! Scrapes and extracts text content from websites redirected by DuckDuckGo !ducky queries.

use crate::cognition::knowledge::KnowledgeIndex;
use crate::constants::tram_scraper::{
    MAX_PARAGRAPH_LENGTH, MIN_ENGLISH_RATIO, MIN_SENTENCE_LENGTH,
};
#[cfg(feature = "net")]
use crate::constants::tram_web_fetch::{
    CITATION_ARTIFACT_REGEX, CITATION_BRACKET_REGEX, CITATION_REFERENCE_REGEX, HTML2TEXT_WIDTH,
    MALFORMED_CITATION_REGEX, MAX_LINKED_URLS_TO_FOLLOW, SCRAPER_FILTER_DOMAINS,
    STANDALONE_CITATION_REGEX, TEXT_FILTER_DOMAINS, TEXT_FILTER_PATTERNS, URL_FILTER_DOMAINS,
    URL_FILTER_KEYWORDS, URL_FILTER_PATHS, URL_FILTER_SCHEMES,
};
use regex::Regex;
use scraper::{Html, Selector};

/// Scrapes text content from a URL and ingests it into the knowledge index.
///
/// # Arguments
///
/// * `url` - The URL to scrape.
/// * `index` - The knowledge index to ingest content into.
/// * `source_label` - Label for the source (e.g., "web-scrape").
///
/// # Returns
///
/// Number of chunks ingested.
#[cfg(feature = "net")]
pub async fn scrape_and_ingest(url: &str, index: &mut KnowledgeIndex, source_label: &str) -> usize {
    let url_lower = url.to_ascii_lowercase();

    if url_lower.contains("youtube.com") || url_lower.contains("youtu.be") {
        return 0;
    }

    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .ok();

    let Some(client) = client else {
        return 0;
    };

    let response = match client.get(url).send().await {
        Ok(r) => r,
        Err(_) => return 0,
    };

    let _final_url = response.url().to_string();
    let text = match response.text().await {
        Ok(t) => t,
        Err(_) => return 0,
    };

    extract_and_ingest(&text, index, source_label)
}

/// Scrapes text content from a URL with redirect handling and ingests it into the knowledge index.
///
/// This function follows HTTP redirects (like DuckDuckGo's internal redirects) to reach the
/// actual external website, then extracts text content using html2text for better readability.
///
/// # Arguments
///
/// * `url` - The URL to scrape (may be a redirect URL).
/// * `index` - The knowledge index to ingest content into.
/// * `source_label` - Label for the source (e.g., "web-deep-scrape").
/// * `topic` - The topic to use for paragraph-level relevance filtering.
///
/// # Returns
///
/// Number of chunks ingested, plus any URLs found in the content for further scraping.
#[cfg(feature = "net")]
pub async fn scrape_and_ingest_with_redirect(
    url: &str,
    index: &mut KnowledgeIndex,
    source_label: &str,
    topic: &str,
) -> (usize, Vec<String>) {
    let url_lower = url.to_ascii_lowercase();

    if SCRAPER_FILTER_DOMAINS
        .iter()
        .any(|&d| url_lower.contains(d))
    {
        return (0, Vec::new());
    }

    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .ok();

    let Some(client) = client else {
        return (0, Vec::new());
    };

    let response = match client.get(url).send().await {
        Ok(r) => r,
        Err(_) => return (0, Vec::new()),
    };

    let final_url = response.url().to_string();
    let html = match response.text().await {
        Ok(t) => t,
        Err(_) => return (0, Vec::new()),
    };

    let text = html2text::from_read(html.as_bytes(), HTML2TEXT_WIDTH).unwrap_or_default();

    let relevant_text = extract_relevant_paragraphs(&text, topic);

    let cleaned = clean_text(&relevant_text);
    if cleaned.split_whitespace().count() < crate::constants::tram_scraper::MIN_SENTENCE_LENGTH {
        return (0, Vec::new());
    }

    let total_ingested = if source_label == "web-deep-scrape" {
        index.ingest_paragraphs(source_label, &cleaned)
    } else {
        let chunk_size = MAX_PARAGRAPH_LENGTH;
        let chunks = split_into_chunks(&cleaned, chunk_size);
        let mut ingested = 0;
        for chunk in chunks {
            ingested += index.ingest_text(source_label, &chunk);
        }
        ingested
    };

    let urls = extract_urls_from_html(&html, &final_url);

    (total_ingested, urls)
}

/// Extracts clean text from HTML and ingests it into the knowledge index.
///
/// # Arguments
///
/// * `html` - Raw HTML string.
/// * `index` - The knowledge index to ingest content into.
/// * `source_label` - Label for the source.
///
/// # Returns
///
/// Number of chunks ingested.
#[cfg(feature = "net")]
pub fn extract_and_ingest(html: &str, index: &mut KnowledgeIndex, source_label: &str) -> usize {
    let document = Html::parse_document(html);

    let mut total_ingested = 0;

    let selector = Selector::parse("p, article, main, .content, .post-content, .entry-content")
        .unwrap_or_else(|_| Selector::parse("p").unwrap());

    for element in document.select(&selector) {
        let text = element.text().collect::<String>();
        let cleaned = clean_text(&text);

        if cleaned.split_whitespace().count() >= MIN_SENTENCE_LENGTH && is_english(&cleaned) {
            let chunks = split_into_chunks(&cleaned, MAX_PARAGRAPH_LENGTH);
            for chunk in chunks {
                total_ingested += index.ingest_text(source_label, &chunk);
            }
        }
    }

    if total_ingested == 0 {
        let fallback_selector = Selector::parse("body").unwrap();
        for element in document.select(&fallback_selector) {
            let text = element.text().collect::<String>();
            let cleaned = clean_text(&text);

            if cleaned.split_whitespace().count() >= MIN_SENTENCE_LENGTH && is_english(&cleaned) {
                let chunks = split_into_chunks(&cleaned, MAX_PARAGRAPH_LENGTH);
                for chunk in chunks {
                    total_ingested += index.ingest_text(source_label, &chunk);
                }
            }
        }
    }

    total_ingested
}

/// Checks if text is primarily English (ASCII characters).
fn is_english(text: &str) -> bool {
    let ascii_chars = text.chars().filter(|c| c.is_ascii()).count();
    let total_chars = text.chars().count();
    if total_chars == 0 {
        return false;
    }
    let ratio = ascii_chars as f64 / total_chars as f64;
    ratio >= MIN_ENGLISH_RATIO && !contains_json(text)
}

/// Checks if text contains JSON-like patterns.
fn contains_json(text: &str) -> bool {
    text.contains("{") && text.contains("}") && (text.contains("\"") || text.contains(":"))
}

/// Extracts paragraphs relevant to the topic by keyword matching.
#[cfg(feature = "net")]
fn extract_relevant_paragraphs(text: &str, topic: &str) -> String {
    let topic_keywords: Vec<String> = topic
        .split_whitespace()
        .filter(|w| w.len() >= MIN_SENTENCE_LENGTH)
        .map(|w| w.to_ascii_lowercase())
        .collect();

    let relevant_paragraphs: Vec<&str> = text
        .split("\n\n")
        .filter(|para| {
            let para_lower = para.to_ascii_lowercase();
            topic_keywords.iter().any(|k| para_lower.contains(k))
        })
        .collect();

    if relevant_paragraphs.is_empty()
        || relevant_paragraphs.join(" ").split_whitespace().count() < 50
    {
        text.to_string()
    } else {
        relevant_paragraphs.join("\n\n")
    }
}

/// Extracts external URLs from HTML anchor tags.
#[cfg(feature = "net")]
fn extract_urls_from_html(html: &str, base_url: &str) -> Vec<String> {
    let document = Html::parse_document(html);
    let selector = Selector::parse("a[href]").unwrap();

    let base = url::Url::parse(base_url)
        .unwrap_or_else(|_| url::Url::parse("https://example.com").unwrap());

    document
        .select(&selector)
        .filter_map(|element| element.value().attr("href"))
        .filter_map(|href| {
            if let Ok(full_url) = base.join(href) {
                Some(full_url.to_string())
            } else {
                None
            }
        })
        .filter(|url| {
            let url_lower = url.to_ascii_lowercase();
            if URL_FILTER_SCHEMES.iter().any(|&s| url_lower.starts_with(s)) {
                return false;
            }
            if !url_lower.starts_with("http://") && !url_lower.starts_with("https://") {
                return false;
            }
            !URL_FILTER_DOMAINS.iter().any(|&d| url.contains(d))
                && !URL_FILTER_PATHS.iter().any(|&p| url_lower.contains(p))
                && !URL_FILTER_KEYWORDS.iter().any(|&k| url_lower.contains(k))
        })
        .take(MAX_LINKED_URLS_TO_FOLLOW)
        .collect()
}

/// Cleans text by removing extra whitespace, script/style content, and Wikipedia artifacts.
#[cfg(feature = "net")]
fn clean_text(text: &str) -> String {
    let citation_regex = Regex::new(CITATION_REFERENCE_REGEX).unwrap();
    let text = citation_regex.replace_all(text, "");

    let artifact_regex = Regex::new(CITATION_ARTIFACT_REGEX).unwrap();
    let text = artifact_regex.replace_all(&text, "");

    let standalone_regex = Regex::new(STANDALONE_CITATION_REGEX).unwrap();
    let text = standalone_regex.replace_all(&text, "");

    let bracket_regex = Regex::new(CITATION_BRACKET_REGEX).unwrap();
    let text = bracket_regex.replace_all(&text, "");

    let malformed_regex = Regex::new(MALFORMED_CITATION_REGEX).unwrap();
    let text = malformed_regex.replace_all(&text, "");

    let text = text.replace(" }", "").replace(" )", "");

    text.split("\n\n")
        .map(|para| {
            para.lines()
                .map(|line| line.trim())
                .filter(|line| {
                    let line_lower = line.to_ascii_lowercase();
                    !line.is_empty()
                        && !TEXT_FILTER_PATTERNS.iter().any(|&p| line.contains(p))
                        && !TEXT_FILTER_DOMAINS.iter().any(|&d| line_lower.contains(d))
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|para| !para.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Cleans text by removing extra whitespace, script/style content, and Wikipedia artifacts.
#[cfg(not(feature = "net"))]
fn clean_text(text: &str) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extracts paragraphs relevant to the topic by keyword matching.
#[cfg(not(feature = "net"))]
fn extract_relevant_paragraphs(text: &str, _topic: &str) -> String {
    text.to_string()
}

/// Extracts external URLs from HTML anchor tags.
#[cfg(not(feature = "net"))]
fn extract_urls_from_html(_html: &str, _base_url: &str) -> Vec<String> {
    Vec::new()
}

/// Splits text into chunks of maximum length.
fn split_into_chunks(text: &str, max_len: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_len = 0;

    for word in words {
        if current_len + word.len() + 1 > max_len && !current_chunk.is_empty() {
            chunks.push(current_chunk.join(" "));
            current_chunk.clear();
            current_len = 0;
        }
        current_chunk.push(word);
        current_len += word.len() + 1;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join(" "));
    }

    chunks
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
