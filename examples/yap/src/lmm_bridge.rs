// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::knowledge::KnowledgeIndex;
use lmm::net::{SearchAggregator, corpus_from_response};
use lmm::tram::DEFAULT_CF_THRESHOLD;
use lmm::tram::{TramConfigBuilder, TramPipeline};

fn domain_key(domain: &str) -> &str {
    domain
        .split_whitespace()
        .find(|w| {
            let lo = w.to_lowercase();
            w.len() >= 4
                && !lo.eq("the")
                && !lo.eq("this")
                && !lo.eq("that")
                && !lo.eq("these")
                && !lo.eq("those")
                && !lo.eq("is")
                && !lo.eq("are")
                && !lo.eq("was")
                && !lo.eq("were")
        })
        .unwrap_or_else(|| domain.split_whitespace().next().unwrap_or("knowledge"))
}

fn build_search_query(topic: &str, domain: &str, turn: u64) -> String {
    if turn <= 1 {
        return topic.to_string();
    }
    let key = domain_key(domain).to_lowercase();
    let lower_topic = topic.to_lowercase();
    if lower_topic.starts_with(&key) && topic.len() > key.len() + 1 {
        let concept = topic[key.len()..].trim();
        if concept.split_whitespace().count() >= 2 {
            return concept.to_string();
        }
    }
    topic.to_string()
}

fn remove_disambiguation_snippets(corpus: &str) -> String {
    const DISAMBIG_MARKERS: &[&str] = &[
        "(band)",
        "(film)",
        "(song)",
        "(album)",
        "(TV series)",
        "(TV show)",
        "(game)",
        "(novel)",
        "(book)",
        "(comics)",
        "(character)",
        "(video game)",
        "(disambiguation)",
    ];
    corpus
        .split(". ")
        .filter(|s| !DISAMBIG_MARKERS.iter().any(|&m| s.contains(m)))
        .collect::<Vec<_>>()
        .join(". ")
}

pub async fn tram_query(topic: &str, domain: &str, turn: u64) -> String {
    let topic = topic.trim();

    let search_query = build_search_query(topic, domain, turn);

    let aggregator = SearchAggregator::new();

    let raw_a = aggregator
        .get_response(&search_query)
        .await
        .map(|r| corpus_from_response(&r))
        .unwrap_or_default();

    let corpus_a = remove_disambiguation_snippets(&raw_a);

    let corpus = if corpus_a.trim().is_empty() {
        let question = format!("what is {search_query}");
        let raw_b = aggregator
            .get_response(&question)
            .await
            .map(|r| corpus_from_response(&r))
            .unwrap_or_default();
        remove_disambiguation_snippets(&raw_b)
    } else {
        corpus_a
    };

    let mut kb = KnowledgeIndex::new();
    kb.ingest_text("search", &corpus);

    let config = TramConfigBuilder::default()
        .no_web(true)
        .think(false)
        .inference_depth(3)
        .cf_threshold(DEFAULT_CF_THRESHOLD)
        .min_kb_words(15)
        .min_prose_words(30)
        .rule_source_limit(12)
        .use_assembly(false)
        .use_summarizer_fallback(true)
        .probability(0.0)
        .build();

    let pipeline = TramPipeline::new(config);
    match pipeline.run(topic, &mut kb).await {
        Ok(result) => {
            if !result.final_text.trim().is_empty() {
                result.final_text
            } else {
                corpus
            }
        }
        Err(_) => corpus,
    }
}
