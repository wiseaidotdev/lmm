// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # Contextual Reordering
//!
//! Reorders sentences based on question intent logic.

use crate::cognition::thought::classifier::ReasoningMode;
use crate::constants::contextual_reorder::MIN_ENTITIES_FOR_ANALOGICAL;
use crate::generation::relevance::RelevanceScorer;

pub struct ContextualReorder;

impl ContextualReorder {
    pub fn reorder(sentences: &[String], query: &str, mode: ReasoningMode) -> Vec<String> {
        match mode {
            ReasoningMode::Evaluative => Self::reorder_evaluative(sentences, query),
            ReasoningMode::Causal => Self::reorder_causal(sentences, query),
            ReasoningMode::Analogical => Self::reorder_analogical(sentences, query),
            ReasoningMode::Definitional => Self::reorder_definitional(sentences, query),
            ReasoningMode::Procedural => Self::reorder_procedural(sentences, query),
            _ => Self::reorder_by_relevance(sentences, query, mode),
        }
    }

    fn reorder_evaluative(sentences: &[String], query: &str) -> Vec<String> {
        let lower_query = query.to_ascii_lowercase();
        let wants_disadvantages = lower_query.contains("disadvantage")
            || lower_query.contains("drawback")
            || lower_query.contains("weakness")
            || lower_query.contains("limitation");

        let mut pros: Vec<String> = Vec::new();
        let mut cons: Vec<String> = Vec::new();
        let mut neutral: Vec<String> = Vec::new();

        for sentence in sentences {
            let lower = sentence.to_ascii_lowercase();
            let is_pro = lower.contains("advantage")
                || lower.contains("benefit")
                || lower.contains("strength")
                || lower.contains("positive");
            let is_con = lower.contains("disadvantage")
                || lower.contains("drawback")
                || lower.contains("weakness")
                || lower.contains("negative");

            if is_pro {
                pros.push(sentence.clone());
            } else if is_con {
                cons.push(sentence.clone());
            } else {
                neutral.push(sentence.clone());
            }
        }

        let mut result = Vec::new();
        if wants_disadvantages {
            result.extend(cons);
            result.extend(neutral);
            result.extend(pros);
        } else {
            result.extend(pros);
            result.extend(neutral);
            result.extend(cons);
        }

        if result.is_empty() {
            sentences.to_vec()
        } else {
            result
        }
    }

    fn reorder_causal(sentences: &[String], query: &str) -> Vec<String> {
        let query_terms = RelevanceScorer::extract_terms(query);

        let mut with_causal: Vec<(String, f64)> = Vec::new();
        let mut without_causal: Vec<(String, f64)> = Vec::new();

        for sentence in sentences {
            let lower = sentence.to_ascii_lowercase();
            let has_causal = lower.contains("because")
                || lower.contains("since")
                || lower.contains("due to")
                || lower.contains("reason")
                || lower.contains("cause")
                || lower.contains("leads to")
                || lower.contains("results in");

            let score =
                RelevanceScorer::score_sentence(sentence, &query_terms, ReasoningMode::Causal);

            if has_causal {
                with_causal.push((sentence.clone(), score));
            } else {
                without_causal.push((sentence.clone(), score));
            }
        }

        with_causal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        without_causal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result: Vec<String> = with_causal.into_iter().map(|(s, _)| s).collect();
        result.extend(without_causal.into_iter().map(|(s, _)| s));

        if result.is_empty() {
            sentences.to_vec()
        } else {
            result
        }
    }

    fn reorder_analogical(sentences: &[String], query: &str) -> Vec<String> {
        let lower_query = query.to_ascii_lowercase();
        let entities = Self::extract_entities(&lower_query);

        if entities.len() < MIN_ENTITIES_FOR_ANALOGICAL {
            return Self::reorder_by_relevance(sentences, query, ReasoningMode::Analogical);
        }

        let entity_a = &entities[0];
        let entity_b = &entities[1];

        let mut sentences_a: Vec<String> = Vec::new();
        let mut sentences_b: Vec<String> = Vec::new();
        let mut sentences_comparison: Vec<String> = Vec::new();
        let mut neutral: Vec<String> = Vec::new();

        for sentence in sentences {
            let lower = sentence.to_ascii_lowercase();
            let mentions_a = lower.contains(entity_a);
            let mentions_b = lower.contains(entity_b);
            let is_comparison = lower.contains("versus")
                || lower.contains("compared")
                || lower.contains("difference")
                || lower.contains("similar")
                || lower.contains("unlike")
                || lower.contains("whereas");

            if mentions_a && mentions_b {
                sentences_comparison.push(sentence.clone());
            } else if mentions_a {
                sentences_a.push(sentence.clone());
            } else if mentions_b {
                sentences_b.push(sentence.clone());
            } else if is_comparison {
                sentences_comparison.push(sentence.clone());
            } else {
                neutral.push(sentence.clone());
            }
        }

        let mut result = Vec::new();
        result.extend(sentences_comparison);
        result.extend(sentences_a);
        result.extend(sentences_b);
        result.extend(neutral);

        if result.is_empty() {
            sentences.to_vec()
        } else {
            result
        }
    }

    fn reorder_definitional(sentences: &[String], query: &str) -> Vec<String> {
        let query_terms = RelevanceScorer::extract_terms(query);

        let mut with_definition: Vec<(String, f64)> = Vec::new();
        let mut without_definition: Vec<(String, f64)> = Vec::new();

        for sentence in sentences {
            let lower = sentence.to_ascii_lowercase();
            let has_definition = lower.contains("definition")
                || lower.contains("defined")
                || lower.contains("refers")
                || lower.contains("means")
                || lower.contains("is a")
                || lower.contains("is an");

            let score = RelevanceScorer::score_sentence(
                sentence,
                &query_terms,
                ReasoningMode::Definitional,
            );

            if has_definition {
                with_definition.push((sentence.clone(), score));
            } else {
                without_definition.push((sentence.clone(), score));
            }
        }

        with_definition.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        without_definition
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result: Vec<String> = with_definition.into_iter().map(|(s, _)| s).collect();
        result.extend(without_definition.into_iter().map(|(s, _)| s));

        if result.is_empty() {
            sentences.to_vec()
        } else {
            result
        }
    }

    fn reorder_procedural(sentences: &[String], query: &str) -> Vec<String> {
        let query_terms = RelevanceScorer::extract_terms(query);

        let mut with_procedural: Vec<(String, f64)> = Vec::new();
        let mut without_procedural: Vec<(String, f64)> = Vec::new();

        for sentence in sentences {
            let lower = sentence.to_ascii_lowercase();
            let has_procedural = lower.contains("step")
                || lower.contains("first")
                || lower.contains("second")
                || lower.contains("then")
                || lower.contains("next")
                || lower.contains("finally")
                || lower.contains("to ")
                || lower.contains("method");

            let score =
                RelevanceScorer::score_sentence(sentence, &query_terms, ReasoningMode::Procedural);

            if has_procedural {
                with_procedural.push((sentence.clone(), score));
            } else {
                without_procedural.push((sentence.clone(), score));
            }
        }

        with_procedural.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        without_procedural
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result: Vec<String> = with_procedural.into_iter().map(|(s, _)| s).collect();
        result.extend(without_procedural.into_iter().map(|(s, _)| s));

        if result.is_empty() {
            sentences.to_vec()
        } else {
            result
        }
    }

    fn reorder_by_relevance(sentences: &[String], query: &str, mode: ReasoningMode) -> Vec<String> {
        let scored = RelevanceScorer::rank_sentences(sentences, query, mode);
        scored.into_iter().map(|(s, _)| s).collect()
    }

    pub fn extract_entities(query: &str) -> Vec<String> {
        let mut entities = Vec::new();

        if let Some(pos) = query.find(" vs ") {
            entities.push(query[..pos].trim().to_string());
            entities.push(query[pos + 4..].trim().to_string());
        } else if let Some(pos) = query.find(" versus ") {
            entities.push(query[..pos].trim().to_string());
            entities.push(query[pos + 8..].trim().to_string());
        } else if let Some(pos) = query.find(" and ") {
            entities.push(query[..pos].trim().to_string());
            entities.push(query[pos + 5..].trim().to_string());
        }

        entities
    }
}
