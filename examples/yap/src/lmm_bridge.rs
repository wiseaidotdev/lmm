// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::knowledge::KnowledgeIndex;
use lmm::expert::inference::InferenceEngine;
use lmm::expert::rule::RuleBase;
use lmm::generation::prose::ProseBuilder;
use lmm::net::{SearchAggregator, corpus_from_response};
use lmm::text::{ParagraphGenerator, TextSummarizer};

fn topic_seed(s: &str) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

fn local_fallback(topic: &str) -> String {
    match ParagraphGenerator::new(2, 60, 5).generate(topic) {
        Ok(p) => p,
        Err(_) => format!(
            "The topic of {topic} spans many dimensions of knowledge and continues \
             to be an active area of research and practical application."
        ),
    }
}

pub async fn tram_query(topic: &str) -> String {
    let topic = topic.trim();
    let aggregator = SearchAggregator::new();

    let api_result = aggregator.get_response(topic).await;

    let corpus = api_result
        .as_ref()
        .map(corpus_from_response)
        .unwrap_or_default();

    if corpus.trim().is_empty() {
        return local_fallback(topic);
    }

    let mut kb = KnowledgeIndex::new();
    kb.ingest_text("search", &corpus);

    let top_chunks: Vec<&str> = kb
        .query(topic, 8)
        .into_iter()
        .map(|c| c.text.as_str())
        .collect();

    let rule_sources: Vec<&str> = top_chunks.iter().copied().take(12).collect();
    let base = RuleBase::from_chunks(&rule_sources);

    let engine = InferenceEngine::new(3, 0.15);
    let seed = topic_seed(topic);

    if let Some(inference) = engine.reason(topic, &base) {
        let prose = ProseBuilder::new(2, 4).build(&inference, &top_chunks, seed);
        if !prose.trim().is_empty() {
            return prose;
        }
    }

    match TextSummarizer::new(4, 60, 5).summarize_with_query(&corpus, topic) {
        Ok(sents) if !sents.is_empty() => sents.join(" "),
        _ => local_fallback(topic),
    }
}

pub fn summarize_to_topic(text: &str) -> String {
    let clean = text.trim();
    if clean.is_empty() {
        return "artificial intelligence".to_string();
    }

    let short = match TextSummarizer::new(1, 40, 3).summarize(clean) {
        Ok(sents) if !sents.is_empty() => sents[0].clone(),
        _ => clean
            .split_whitespace()
            .take(12)
            .collect::<Vec<_>>()
            .join(" "),
    };

    let topic: String = short
        .split_whitespace()
        .filter(|w| w.len() >= 4)
        .take(5)
        .collect::<Vec<_>>()
        .join(" ");

    if topic.is_empty() {
        "knowledge and reasoning".to_string()
    } else {
        topic
    }
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
