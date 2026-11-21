// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Text generation methods for [`LmmAgent`].
//!
//! Covers [`generate`](LmmAgent::generate), [`search`](LmmAgent::search),
//! and the private helpers `best_sentence`, `symbolic_continuation`, and
//! `domain_seed`.

use anyhow::Result;
use lmm::cognition::types::Message;
use lmm::predict::TextPredictor;
#[cfg(feature = "net")]
use std::collections::HashSet;

#[cfg(feature = "net")]
use duckduckgo::browser::Browser;
#[cfg(feature = "net")]
use duckduckgo::user_agents::get as get_ua;

use super::LmmAgent;

impl LmmAgent {
    /// Generates a textual response to `request` using [`lmm::predict::TextPredictor`].
    ///
    /// `TextPredictor` fits a tone trajectory and a rhythm trajectory over the
    /// input tokens using symbolic regression, then selects continuation words
    /// from compile-time lexical pools: entirely deterministic, no LLM API
    /// required.
    ///
    /// When the `net` feature is enabled, the seed is enriched with DuckDuckGo
    /// search snippets before feeding it to the predictor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[tokio::main]
    /// async fn main() {
    ///     use lmm_agent::agent::LmmAgent;
    ///     let mut agent = LmmAgent::new("Tester".into(), "Rust is fast.".into());
    ///     let result = agent.generate("the universe reveals its truth").await;
    ///     assert!(result.is_ok());
    ///     assert!(!result.unwrap().is_empty());
    /// }
    /// ```
    pub async fn generate(&mut self, request: &str) -> Result<String> {
        if !self.knowledge_index.is_empty()
            && let Some(answer) = self.knowledge_index.answer(request, 5)
        {
            self.add_message(Message::new("user", request.to_string()));
            self.add_message(Message::new("assistant", answer.clone()));
            return Ok(answer);
        }

        #[cfg(feature = "net")]
        let result = {
            let corpus = self.search(request, 5).await.unwrap_or_default();
            if let Some(sentence) = Self::best_sentence(&corpus, request) {
                sentence
            } else {
                let seed = if corpus.is_empty() {
                    Self::domain_seed(request, &self.behavior)
                } else {
                    format!("{request} {corpus}")
                };
                Self::symbolic_continuation(seed)
            }
        };

        #[cfg(not(feature = "net"))]
        let result = {
            let seed = Self::domain_seed(request, &self.behavior);
            Self::symbolic_continuation(seed)
        };

        self.add_message(Message::new("user", request.to_string()));
        self.add_message(Message::new("assistant", result.clone()));
        Ok(result)
    }

    /// Builds a domain-biased seed string from the request and agent behavior.
    fn domain_seed(request: &str, behavior: &str) -> String {
        const STOP: &[&str] = &[
            "a", "an", "the", "and", "or", "of", "to", "in", "is", "are", "be", "for", "on", "at",
            "by", "as", "it", "its",
        ];
        let domain_words: Vec<&str> = behavior
            .split_whitespace()
            .filter(|w| {
                let lw = w.to_ascii_lowercase();
                !STOP.contains(&lw.as_str()) && w.len() > 3
            })
            .take(6)
            .collect();

        let mut seed = request.to_string();
        if !domain_words.is_empty() {
            seed.push(' ');
            seed.push_str(&domain_words.join(" "));
        }
        if seed.split_whitespace().count() < 2 {
            seed.push_str(" and");
        }
        seed
    }

    /// Runs the symbolic predictor on a seed and returns the continuation.
    fn symbolic_continuation(seed: String) -> String {
        let mut predictor = TextPredictor::new(20, 40, 3);
        if let Ok(lex) = lmm::lexicon::Lexicon::load_system() {
            predictor = predictor.with_lexicon(lex);
        }
        predictor
            .predict_continuation(&seed, 120)
            .map(|c| format!("{} {}", seed.trim(), c.continuation.trim()))
            .unwrap_or(seed)
    }

    /// Returns the sentence from `corpus` with the highest token overlap with `query`.
    ///
    /// Returns `None` if no sentence has meaningful overlap.
    #[cfg(feature = "net")]
    fn best_sentence(corpus: &str, query: &str) -> Option<String> {
        let query_tokens: HashSet<String> = query
            .split_whitespace()
            .map(|w| w.to_ascii_lowercase())
            .collect();

        corpus
            .split(['.', '!', '?'])
            .map(str::trim)
            .filter(|s| s.split_whitespace().count() >= 5)
            .map(|sentence| {
                let sentence_tokens: HashSet<String> = sentence
                    .split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .collect();
                let overlap = query_tokens.intersection(&sentence_tokens).count();
                (overlap, sentence.to_string())
            })
            .filter(|(overlap, _)| *overlap >= 2)
            .max_by_key(|(overlap, _)| *overlap)
            .map(|(_, sentence)| sentence)
    }

    /// Searches DuckDuckGo for `query` (requires `net` feature).
    #[cfg(feature = "net")]
    pub async fn search(&self, query: &str, limit: usize) -> Result<String> {
        let browser = Browser::new();
        let ua = get_ua("firefox").unwrap_or("Mozilla/5.0");
        let results = browser.lite_search(query, "wt-wt", Some(limit), ua).await?;

        let corpus = results
            .iter()
            .filter_map(|r| {
                let snippet = r.snippet.trim();
                if !snippet.is_empty() {
                    Some(snippet.to_string())
                } else if !r.title.trim().is_empty() {
                    Some(r.title.trim().to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        Ok(corpus)
    }

    /// No-op search when the `net` feature is disabled.
    #[cfg(not(feature = "net"))]
    pub async fn search(&self, _query: &str, _limit: usize) -> Result<String> {
        Ok(String::new())
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
