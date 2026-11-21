// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Knowledge ingestion and retrieval methods for [`LmmAgent`].
//!
//! Covers [`ingest`](LmmAgent::ingest), [`query_knowledge`](LmmAgent::query_knowledge),
//! and [`answer_from_knowledge`](LmmAgent::answer_from_knowledge).

use anyhow::Result;
use lmm::cognition::knowledge::{KnowledgeSource, ingest as knowledge_ingest};

use super::LmmAgent;

impl LmmAgent {
    /// Ingests a [`KnowledgeSource`] into this agent's [`KnowledgeIndex`].
    ///
    /// Returns the number of new sentence-level chunks added to the index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    /// use lmm::cognition::knowledge::KnowledgeSource;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut agent = LmmAgent::new("KA Agent".into(), "Rust ownership.".into());
    ///     let n = agent
    ///         .ingest(KnowledgeSource::RawText(
    ///             "Rust prevents data races at compile time through its ownership system. \
    ///              The borrow checker enforces these rules statically.".into(),
    ///         ))
    ///         .await
    ///         .unwrap();
    ///     assert!(n > 0);
    /// }
    /// ```
    pub async fn ingest(&mut self, source: KnowledgeSource) -> Result<usize> {
        knowledge_ingest(&mut self.knowledge_index, source).await
    }

    /// Returns the top-`top_k` relevant passages from the knowledge index for `question`.
    ///
    /// Returns an empty `Vec` when the index contains no matching material.
    pub fn query_knowledge(&self, question: &str, top_k: usize) -> Vec<String> {
        self.knowledge_index
            .query(question, top_k)
            .into_iter()
            .map(|c| c.text.clone())
            .collect()
    }

    /// Produces an extractive answer to `question` from the knowledge index.
    ///
    /// Retrieves the top-5 relevant chunks, concatenates them, and runs
    /// [`lmm::text::TextSummarizer`] to select the most informative sentences.
    ///
    /// Returns `None` when the index is empty or no relevant material is found.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    /// use lmm::cognition::knowledge::KnowledgeSource;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut agent = LmmAgent::new("QA Agent".into(), "Rust.".into());
    ///     agent
    ///         .ingest(KnowledgeSource::RawText(
    ///             "Rust prevents data races through ownership. \
    ///              The borrow checker ensures memory safety at compile time.".into(),
    ///         ))
    ///         .await
    ///         .unwrap();
    ///     let answer = agent.answer_from_knowledge("How does Rust handle memory?");
    ///     assert!(answer.is_some());
    /// }
    /// ```
    pub fn answer_from_knowledge(&self, question: &str) -> Option<String> {
        self.knowledge_index.answer(question, 5)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
