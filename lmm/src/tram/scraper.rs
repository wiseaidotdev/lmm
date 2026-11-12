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
use scraper::{Html, Selector};

const MIN_SENTENCE_LENGTH: usize = 8;
const MAX_PARAGRAPH_LENGTH: usize = 500;
const MIN_ENGLISH_RATIO: f64 = 0.7;

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
        .build()
        .ok();

    let Some(client) = client else {
        return 0;
    };

    let response = match client.get(url).send().await {
        Ok(r) => r,
        Err(_) => return 0,
    };

    let text = match response.text().await {
        Ok(t) => t,
        Err(_) => return 0,
    };

    extract_and_ingest(&text, index, source_label)
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

/// Cleans text by removing extra whitespace and script/style content.
fn clean_text(text: &str) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
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
