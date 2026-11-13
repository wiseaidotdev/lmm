use crate::types::SearchLink;
use lmm::cognition::{
    knowledge::KnowledgeIndex, r#loop::ThinkLoop, search::SearchOracle,
    thought::AdaptiveThoughtEngine,
};
use lmm::constants::app::EVALUATIVE_KEYWORDS;
use lmm::constants::tram_answer::{ADVANTAGE_KEYWORDS, HISTORICAL_MARKERS, HISTORY_KEYWORDS};
use lmm::expert::{inference::InferenceEngine, rule::RuleBase};
use lmm::net::{Response, SearchAggregator, corpus_from_response};
use lmm::stochastic::StochasticEnhancer;
use lmm::text::TextSummarizer;
use lmm::tram::{
    DEFAULT_MIN_RULE_COUNT,
    config::{DEFAULT_THINK_PATIENCE, DEFAULT_THINK_STEPS, DEFAULT_THINK_THRESHOLD},
    query::extract_topic,
    response::is_quality_answer,
    split_sentences_with_punct,
};

fn enhance(text: &str, stochastic: bool, prob: f64) -> String {
    if stochastic {
        StochasticEnhancer::new(prob).enhance(text)
    } else {
        text.to_string()
    }
}

fn extract_topic_name(prompt: &str) -> String {
    let topic = extract_topic(prompt);
    topic
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
}

fn derive_kb_query(prompt: &str, topic: &str) -> String {
    let lower = prompt.to_ascii_lowercase();
    let evaluative_word = lower
        .split_whitespace()
        .find(|w| {
            w.len() > 4
                && !matches!(
                    *w,
                    "what"
                        | "does"
                        | "have"
                        | "that"
                        | "this"
                        | "with"
                        | "from"
                        | "into"
                        | "are"
                        | "the"
                        | "about"
                )
        })
        .unwrap_or("");
    if !evaluative_word.is_empty() {
        format!("{} {}", evaluative_word, topic)
    } else {
        topic.to_string()
    }
}

fn extract_links(result: &anyhow::Result<Response>) -> Vec<SearchLink> {
    result
        .as_ref()
        .map(|resp| {
            resp.related_topics
                .iter()
                .filter_map(|t| {
                    let url = t.first_url.clone()?;
                    if url.is_empty() {
                        return None;
                    }
                    let text = t.text.clone().unwrap_or_default();
                    let title: String = text
                        .split_whitespace()
                        .take(6)
                        .collect::<Vec<_>>()
                        .join(" ");
                    let title = if title.is_empty() {
                        url.clone()
                    } else {
                        format!("{}...", title)
                    };
                    Some(SearchLink { title, url })
                })
                .take(10)
                .collect()
        })
        .unwrap_or_default()
}

async fn wasm_prefetch_web(
    prompt: &str,
    topic: &str,
    index: &mut KnowledgeIndex,
) -> Vec<SearchLink> {
    let aggregator = SearchAggregator::new();
    let lower = prompt.to_ascii_lowercase();
    let mut all_links = Vec::new();

    let mut seen_urls = std::collections::HashSet::new();

    let base_queries = vec![topic.to_string(), format!("{} programming language", topic)];

    for query in &base_queries {
        let result = aggregator.get_response(query).await;
        let corpus = result
            .as_ref()
            .map(|r| corpus_from_response(r))
            .unwrap_or_default();
        let _ingested = index.ingest_text("web", &corpus);

        let links = extract_links(&result);
        for link in links {
            if seen_urls.insert(link.url.clone()) {
                all_links.push(link);
            }
        }
    }

    if lower.contains("advantages") || lower.contains("benefits") {
        let intent_queries = vec![
            format!("advantages of {}", topic),
            format!("benefits of {}", topic),
            format!("disadvantages of {}", topic),
            format!("pros and cons of {}", topic),
        ];

        for query in &intent_queries {
            let result = aggregator.get_response(query).await;
            let corpus = result
                .as_ref()
                .map(|r| corpus_from_response(r))
                .unwrap_or_default();
            let _ingested = index.ingest_text("web", &corpus);

            let links = extract_links(&result);
            for link in links {
                if seen_urls.insert(link.url.clone()) {
                    all_links.push(link);
                }
            }
        }
    }

    all_links
}

pub async fn tram_query(
    topic: &str,
    think: bool,
    deep: bool,
    iterations: usize,
    inference_depth: usize,
    cf_threshold: f64,
    stochastic: bool,
    probability: f64,
) -> (String, Vec<SearchLink>) {
    let prompt = topic.trim();
    let extracted_topic = extract_topic(prompt);
    let mut index = KnowledgeIndex::new();

    let links = wasm_prefetch_web(prompt, &extracted_topic, &mut index).await;

    let mut oracle = SearchOracle::new(5);

    if !index.is_empty() {
        let topic_query = extract_topic(prompt);
        let kb_corpus: String = index
            .query(&topic_query, 8)
            .into_iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        if !kb_corpus.is_empty() {
            oracle.seed_cache(&topic_query, &kb_corpus);
            oracle.seed_cache(prompt, &kb_corpus);
        }
    }

    let (_ate_goal, cold_corpus) = if think {
        let ate = AdaptiveThoughtEngine::new(
            DEFAULT_THINK_STEPS,
            DEFAULT_THINK_THRESHOLD,
            DEFAULT_THINK_PATIENCE,
        );
        let thought_result = ate.run(prompt, &mut oracle).await;

        if !thought_result.factual_insight.is_empty() {
            index.ingest_text("ate-insight", &thought_result.factual_insight);
        }

        let goal = thought_result
            .prioritized_queries
            .into_iter()
            .next()
            .unwrap_or_else(|| prompt.to_string());

        let convergence_threshold = 0.45;
        let think_iterations = if index.is_empty() {
            iterations
        } else {
            iterations.min(6)
        };
        let mut lp = ThinkLoop::builder(goal.clone())
            .max_iterations(think_iterations)
            .convergence_threshold(convergence_threshold)
            .k_proportional(1.2)
            .k_integral(0.08)
            .build();

        let _think_result = lp.run(&mut oracle).await;

        for entry in lp.cold.all() {
            index.ingest_text("think", &entry.content);
        }

        let cold = lp
            .cold
            .all()
            .iter()
            .map(|e| e.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        (goal, cold)
    } else {
        (prompt.to_string(), String::new())
    };

    let topic = extract_topic(prompt);
    let kb_query = derive_kb_query(prompt, &topic);
    let top_k = 6;

    let prompt_lower = prompt.to_ascii_lowercase();
    let is_evaluative = EVALUATIVE_KEYWORDS
        .iter()
        .any(|&keyword| prompt_lower.contains(keyword));

    let (kb_chunks, combined_answer): (Vec<String>, Option<String>) = if deep && is_evaluative {
        let chunks: Vec<String> = index
            .query(&kb_query, top_k)
            .into_iter()
            .map(|c| c.text.clone())
            .collect();

        let filtered_chunks: Vec<&str> = chunks
            .iter()
            .map(String::as_str)
            .filter(|s| {
                let words = s.split_whitespace().count();
                let lower = s.to_ascii_lowercase();
                words >= 10
                    && !lower.contains("code")
                    && !lower.contains("function")
                    && !lower.contains(";")
                    && !lower.contains("[]")
            })
            .collect();

        let answer = index.answer(&kb_query, top_k);

        let combined = match (&answer, filtered_chunks.len()) {
            (Some(ia), n) if n >= 2 => Some(format!("{}\n\n{}", ia, filtered_chunks.join("\n\n"))),
            _ => answer,
        };

        (chunks, combined)
    } else {
        let mut chunks: Vec<String> = index
            .query(&kb_query, top_k)
            .into_iter()
            .map(|c| c.text.clone())
            .collect();

        let answer = if chunks.is_empty() {
            chunks = index
                .query(&topic, top_k)
                .into_iter()
                .map(|c| c.text.clone())
                .collect();
            if chunks.is_empty() {
                chunks = index
                    .query(prompt, top_k)
                    .into_iter()
                    .map(|c| c.text.clone())
                    .collect();
                index.answer(prompt, top_k)
            } else {
                index.answer(&topic, top_k)
            }
        } else {
            index.answer(&kb_query, top_k)
        };

        (chunks, answer)
    };

    let answer_text = match combined_answer {
        Some(answer) if is_quality_answer(&answer) => {
            if think && !deep {
                let prompt_asks_history = HISTORY_KEYWORDS
                    .iter()
                    .any(|&keyword| prompt_lower.contains(keyword));

                let filtered_answer = if !prompt_asks_history {
                    let split_result = split_sentences_with_punct(&answer);
                    let filtered: Vec<String> = split_result
                        .into_iter()
                        .filter(|s| {
                            let lower = s.to_ascii_lowercase();
                            let has_year = lower.chars().any(|c| c.is_ascii_digit());
                            let has_historical_marker = HISTORICAL_MARKERS
                                .iter()
                                .any(|&marker| lower.contains(marker));
                            !(has_year && has_historical_marker)
                        })
                        .collect();
                    if filtered.is_empty() {
                        answer.clone()
                    } else {
                        filtered.join(" ")
                    }
                } else {
                    answer.clone()
                };

                let prefixed = if ADVANTAGE_KEYWORDS
                    .iter()
                    .any(|&keyword| prompt_lower.contains(keyword))
                {
                    let topic_name = extract_topic_name(prompt);
                    format!(
                        "The advantages of {} are as follows. {}",
                        topic_name, filtered_answer
                    )
                } else {
                    filtered_answer.to_string()
                };

                prefixed
            } else {
                answer.to_string()
            }
        }
        Some(answer) => answer.to_string(),
        None => {
            if !cold_corpus.is_empty() {
                cold_corpus
            } else {
                "I could not find relevant information for your query. Web search may be restricted in the browser environment.".to_string()
            }
        }
    };

    let chunk_refs: Vec<&str> = kb_chunks.iter().map(|s| s.as_str()).collect();
    let rule_base = RuleBase::from_chunks(&chunk_refs);
    let expert_result = if rule_base.rules.len() >= DEFAULT_MIN_RULE_COUNT {
        let engine = InferenceEngine::new(inference_depth, cf_threshold);
        engine.reason(prompt, &rule_base)
    } else {
        None
    };

    let final_text = if deep {
        answer_text
    } else if let Some(result) = expert_result {
        if !result.conclusion.is_empty() {
            result.conclusion
        } else {
            answer_text
        }
    } else {
        answer_text
    };

    let deduplicated = if final_text.len() > 100 {
        let sentences: Vec<&str> = final_text.split('.').collect();
        let mut seen = std::collections::HashSet::new();
        let deduped: Vec<&str> = sentences
            .into_iter()
            .filter(|s| {
                let normalized = s.trim().to_ascii_lowercase();
                seen.insert(normalized)
            })
            .collect();
        deduped.join(".")
    } else {
        final_text
    };

    let response = enhance(&deduplicated, stochastic, probability);

    (response, links)
}

pub fn summarize(text: &str, count: usize, stochastic: bool, prob: f64) -> String {
    match TextSummarizer::new(count.max(1), 50, 5).summarize(text) {
        Ok(sentences) => sentences
            .iter()
            .map(|s| enhance(s, stochastic, prob))
            .collect::<Vec<_>>()
            .join(" "),
        Err(_) => {
            "Input needs complete sentences (ending with . ! ?) of at least 5 words.".to_string()
        }
    }
}
