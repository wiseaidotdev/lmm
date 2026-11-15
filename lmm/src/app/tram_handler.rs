// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Pipeline Handler

#[cfg(feature = "cli")]
use {
    crate::{
        cli::{
            display::{banner, divider},
            tram::{
                assemble_paragraphs_for, derive_framing_topic, prefetch_web_for_query,
                query::extract_topic,
                response::{build_symbolic_response, derive_kb_query, is_quality_answer},
                restructure_analogical_response,
            },
        },
        cognition::{
            ReasoningMode,
            code_responder::CodeResponder,
            knowledge::{KnowledgeIndex, KnowledgeSource, ingest as cognition_ingest},
            r#loop::ThinkLoop,
            response_framer::ResponseFramer,
            search::SearchOracle,
            task_decomposer::{TaskDecomposer, TaskKind},
            thought::AdaptiveThoughtEngine,
            thought::classifier::InputClassifier,
        },
        constants::app::{
            DEFAULT_CODE_LANGUAGE, DEFAULT_STOCHASTIC_PROBABILITY, EVALUATIVE_KEYWORDS,
            VS_SPLITTER_PRIMARY, VS_SPLITTER_SECONDARY,
        },
        constants::stochastic::{PROBABILITY_MAX, PROBABILITY_MIN},
        constants::tram_answer::{
            ADVANTAGE_KEYWORDS, HISTORICAL_MARKERS, HISTORY_KEYWORDS, MIN_FILTERED_PARAGRAPHS,
        },
        constants::tram_scraper::{
            ARTICLE_GAP_PATTERNS, CITATION_SEMICOLON_AND_GAP_PATTERN,
            CITATION_SEMICOLON_GAP_PATTERN, CITATION_SEMICOLON_WAS_GAP_PATTERN, CODE_INDICATORS,
            CODE_KEYWORDS, EMPTY_GAP_PATTERN, INCOMPLETE_COMMON_PATTERN,
            INCOMPLETE_PARENTHETICAL_PATTERNS, INCOMPLETE_QUOTE_EXCEPTION,
            INCOMPLETE_QUOTE_PATTERN, INCOMPLETE_THROUGH_MODULES_PATTERN,
            INCOMPLETE_THROUGH_PATTERN, MAX_DEEP_PARAGRAPHS_TO_SHOW, MIN_DEEP_PARAGRAPH_WORDS,
            MONTH_PATTERNS, SENTENCE_END_INCOMPLETE, STANDALONE_PUNCTUATION, TABLE_INDICATORS,
        },
        expert::{inference::InferenceEngine, rule::RuleBase},
        generation::{dedup::GlobalDeduplicator, phrases_mixer::PhrasesMixer, prose::ProseBuilder},
        stochastic::StochasticEnhancer,
        tram::{DEFAULT_CF_THRESHOLD, DEFAULT_MIN_RULE_COUNT, split_sentences_with_punct},
    },
    std::{
        path::Path,
        time::{SystemTime, UNIX_EPOCH},
    },
    tracing::{error, info},
};

#[cfg(all(feature = "cli", feature = "net"))]
use crate::{
    constants::tram_web_fetch::{DEEP_SEARCH_URL_FOOTER_BULLET, DEEP_SEARCH_URL_FOOTER_HEADER},
};


/// Executes the TRAM (Text Reasoning and Applied Mathematics) pipeline.
///
/// TRAM is the primary reasoning mode of the `lmm` binary, a CPU-only,
/// training-free alternative to GPT-style language models. It operates in
/// five stages:
///
/// ```text
/// prompt
///   │
///   ├─► KnowledgeIndex ingestion  (local KB file or directory)
///   │       ↓ if empty and `net` feature is on
///   │   DuckDuckGo pre-fetch  (10 results ingested directly into KnowledgeIndex)
///   │       ↓
///   │   ThinkLoop::run(&mut oracle)   (PID closed-loop, N iter, threshold 0.25)
///   │       ↓  hot-store sentences written to KnowledgeIndex
///   │   KnowledgeIndex::answer(prompt)  (IDF-ranked extractive answer, low stochastic)
///   │       ↓ if None and cold-memory has content → TextSummarizer on cold corpus
///   │       ↓ if None → ParagraphGenerator (topic-aware symbolic synthesis)
///   │       ↓
///   └─► StochasticEnhancer::enhance_detailed  (guaranteed per-run variation)
/// ```
///
/// No GPU, no neural weights, no API keys. Every response is computed fresh
/// from first principles using symbolic mathematics and stochastic determinism.
///
/// # Arguments
///
/// * `prompt`           - Natural-language question or task description.
/// * `knowledge_base`   - Optional filesystem path to a KB file or directory.
/// * `probability`      - Optional per-word synonym-substitution probability [0.0, 1.0].
/// * `verbose`          - Whether to emit rich diagnostic output.
/// * `iterations`       - Maximum ThinkLoop iterations (default 10).
/// * `inference_depth`  - Backward-chaining depth limit for expert reasoning.
/// * `cf_threshold`     - Minimum certainty factor threshold for expert inference.
/// * `think`            - Whether to run the Adaptive Thought Engine pre-pass (off by default).
/// * `think_steps`      - Maximum thought steps for ATE (only used when `think` is `true`).
/// * `think_threshold`  - DS-belief midpoint early-stop threshold for ATE [0, 1].
/// * `think_patience`   - Consecutive non-improving steps before ATE flags stagnation.
/// * `max_tasks`        - Upper bound on task-decomposer sub-tasks.
/// * `no_frame`         - Skip response framing when `true`.
/// * `deep`             - Activate deep-scrape web enrichment mode.
///
/// # Time Complexity
///
/// O(|KB| + iterations × |oracle|) for the think loop;
/// O(|output|) for the stochastic pass.
///
/// # Space Complexity
///
/// O(|KB| + |hot_store|) for the in-memory knowledge index.
#[cfg(feature = "cli")]
#[allow(clippy::too_many_arguments)]
pub async fn handle_tram(
    prompt: String,
    knowledge_base: Option<String>,
    probability: Option<f64>,
    verbose: bool,
    iterations: usize,
    inference_depth: usize,
    cf_threshold: f64,
    think: bool,
    think_steps: usize,
    think_threshold: f64,
    think_patience: usize,
    max_tasks: usize,
    no_frame: bool,
    deep: bool,
) -> anyhow::Result<()> {
    banner("TRAM · Text Reasoning and Applied Mathematics", "🧮");
    info!("  ❓ Prompt    : {:?}", prompt);
    info!("  🔁 Iterations: {}", iterations);
    info!(
        "  🔍 Inference : depth={} CF≥{:.2}",
        inference_depth, DEFAULT_CF_THRESHOLD
    );

    let mut index = KnowledgeIndex::new();
    let kb_provided = knowledge_base.is_some();

    if let Some(kb_path) = knowledge_base {
        let path = Path::new(&kb_path);
        let source = if path.is_file() {
            Some(KnowledgeSource::File(path.to_path_buf()))
        } else if path.is_dir() {
            Some(KnowledgeSource::Dir(path.to_path_buf()))
        } else {
            error!("  ⚠️  KB path not found: {:?}", kb_path);
            None
        };
        if let Some(src) = source {
            match cognition_ingest(&mut index, src).await {
                Ok(n) => info!("  📚 KB ingested: {} chunks from {:?}", n, kb_path),
                Err(e) => error!("  ⚠️  KB ingestion failed: {e}"),
            }
        }
    }

    divider("Web Knowledge");
    let web_result =
        crate::tram::web_fetch::prefetch_web(&prompt, &mut index, kb_provided, deep, verbose).await;

    let prompt_mode = InputClassifier::classify(&prompt);
    let sub_tasks = TaskDecomposer::decompose_intelligent(&prompt, prompt_mode, max_tasks, 1.0);
    for task in &sub_tasks {
        if let TaskKind::WebSearch(q) = &task.kind {
            prefetch_web_for_query(q, &mut index, verbose).await;
        }
    }

    let run_seed: u64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;

    let mut oracle = SearchOracle::new(5);

    if !index.is_empty() {
        let topic_query = extract_topic(&prompt);
        let kb_corpus: String = index
            .query(&topic_query, 8)
            .into_iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        if !kb_corpus.is_empty() {
            oracle.seed_cache(&topic_query, &kb_corpus);
            oracle.seed_cache(&prompt, &kb_corpus);
        }
    }

    let ate_goal: String = if think {
        divider("Adaptive Thought Engine");

        let ate = AdaptiveThoughtEngine::new(think_steps, think_threshold, think_patience);
        let thought_result = ate.run(&prompt, &mut oracle).await;

        info!(
            "  🧠 ATE: mode={} steps={} confidence={:.3}",
            thought_result.chain.mode.label(),
            thought_result.chain.steps.len(),
            thought_result.confidence
        );

        if verbose {
            for step in &thought_result.chain.steps {
                info!(
                    "  💬 Thought {} [{}]: {}",
                    step.index + 1,
                    step.mode.label(),
                    step.reasoning
                );
            }
        }

        if !thought_result.factual_insight.is_empty() {
            index.ingest_text("ate-insight", &thought_result.factual_insight);
        }

        thought_result
            .prioritized_queries
            .into_iter()
            .next()
            .unwrap_or_else(|| prompt.clone())
    } else {
        prompt.clone()
    };

    divider("ThinkLoop Reasoning");

    let convergence_threshold = 0.45;
    let think_iterations = if index.is_empty() {
        iterations
    } else {
        iterations.min(6)
    };
    let mut lp = ThinkLoop::builder(ate_goal)
        .max_iterations(think_iterations)
        .convergence_threshold(convergence_threshold)
        .k_proportional(1.2)
        .k_integral(0.08)
        .build();

    let think_result = lp.run(&mut oracle).await;
    let think_converged =
        think_result.converged || (!index.is_empty() && think_result.final_error >= 0.99);
    info!(
        "  🔄 ThinkLoop: converged={} steps={} error={:.3}",
        think_converged, think_result.steps, think_result.final_error
    );

    let cold_corpus: String = lp
        .cold
        .all()
        .iter()
        .map(|e| e.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    for entry in lp.cold.all() {
        let chunks_added = index.ingest_text("think", &entry.content);
        if verbose {
            info!("  💡 Memory → KB: +{} chunk(s)", chunks_added);
        }
    }

    divider("Knowledge Retrieval");

    let topic = extract_topic(&prompt);
    let kb_query = derive_kb_query(&prompt, prompt_mode, &topic);
    let top_k = 6 + (run_seed % 5) as usize;

    let prompt_lower = prompt.to_ascii_lowercase();
    let is_evaluative = EVALUATIVE_KEYWORDS
        .iter()
        .any(|&keyword| prompt_lower.contains(keyword));

    let (kb_chunks, combined_answer) = if matches!(prompt_mode, ReasoningMode::Analogical) {
        let topic_lower = topic.to_ascii_lowercase();
        let mut chunks = Vec::new();
        let vs_splitter = if topic_lower.contains(VS_SPLITTER_PRIMARY) {
            VS_SPLITTER_PRIMARY
        } else if topic_lower.contains(VS_SPLITTER_SECONDARY) {
            VS_SPLITTER_SECONDARY
        } else {
            ""
        };

        let combined_answer = if !vs_splitter.is_empty() {
            if let Some(pos) = topic_lower.find(vs_splitter) {
                let entity_a = topic[..pos].trim();
                let entity_b = topic[pos + vs_splitter.len()..].trim();
                let entity_top_k = top_k.max(3);

                info!(
                    "  Analogical retrieval: entity_a='{}', entity_b='{}'",
                    entity_a, entity_b
                );

                chunks.extend(
                    index
                        .query(entity_a, entity_top_k)
                        .into_iter()
                        .map(|c| c.text.clone()),
                );
                chunks.extend(
                    index
                        .query(entity_b, entity_top_k)
                        .into_iter()
                        .map(|c| c.text.clone()),
                );

                let ans_a = index.answer(entity_a, entity_top_k);
                let ans_b = index.answer(entity_b, entity_top_k);
                match (ans_a, ans_b) {
                    (Some(a), Some(b)) => Some(format!("{} {}", a, b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            } else {
                chunks.extend(
                    index
                        .query(&kb_query, top_k)
                        .into_iter()
                        .map(|c| c.text.clone()),
                );
                index.answer(&kb_query, top_k)
            }
        } else {
            chunks.extend(
                index
                    .query(&kb_query, top_k)
                    .into_iter()
                    .map(|c| c.text.clone()),
            );
            index.answer(&kb_query, top_k)
        };
        (chunks, combined_answer)
    } else if is_evaluative {
        let mut chunks = Vec::new();
        chunks.extend(
            index
                .query(&kb_query, top_k)
                .into_iter()
                .map(|c| c.text.clone()),
        );
        chunks.extend(
            index
                .query(&topic, top_k)
                .into_iter()
                .map(|c| c.text.clone()),
        );
        let eval_queries = vec![
            format!("advantages of {}", topic),
            format!("benefits of {}", topic),
            format!("pros and cons of {}", topic),
        ];
        for eq in &eval_queries {
            chunks.extend(index.query(eq, top_k).into_iter().map(|c| c.text.clone()));
        }
        if deep {
            info!("  🔍 Deep mode: querying IA content excluding deep-scraped source");
            let ia_answer = index
                .answer_excluding_source(&eval_queries[0], top_k, "web-deep-scrape")
                .or_else(|| index.answer_excluding_source(&topic, top_k, "web-deep-scrape"));

            let all_deep_chunks = index
                .query_source("web-deep-scrape", top_k * 5)
                .into_iter()
                .map(|c| c.text.clone())
                .collect::<Vec<_>>();
            info!(
                "  🔍 Deep mode: query_source returned {} chunks",
                all_deep_chunks.len()
            );
            chunks.extend(all_deep_chunks.clone());

            let filtered_deep: Vec<String> = all_deep_chunks
                .into_iter()
                .filter(|s| {
                    let words = s.split_whitespace().count();
                    let lower = s.to_ascii_lowercase();
                    if words < MIN_DEEP_PARAGRAPH_WORDS {
                        return false;
                    }
                    if lower.contains(EMPTY_GAP_PATTERN) {
                        return false;
                    }
                    if lower.contains(CITATION_SEMICOLON_GAP_PATTERN)
                        || lower.contains(CITATION_SEMICOLON_AND_GAP_PATTERN)
                        || lower.contains(CITATION_SEMICOLON_WAS_GAP_PATTERN)
                    {
                        return false;
                    }
                    if lower
                        .split_whitespace()
                        .any(|w| w.ends_with(';') && w.len() < 5)
                    {
                        return false;
                    }
                    if lower.contains(INCOMPLETE_THROUGH_PATTERN)
                        || lower.contains(INCOMPLETE_THROUGH_MODULES_PATTERN)
                    {
                        return false;
                    }
                    if lower.starts_with(INCOMPLETE_COMMON_PATTERN) {
                        return false;
                    }
                    for pattern in ARTICLE_GAP_PATTERNS {
                        if lower.contains(pattern) {
                            return false;
                        }
                    }
                    for keyword in CODE_KEYWORDS {
                        if lower.contains(keyword) {
                            return false;
                        }
                    }
                    for indicator in TABLE_INDICATORS {
                        if s.contains(indicator) {
                            return false;
                        }
                    }
                    for indicator in CODE_INDICATORS {
                        if lower.contains(indicator) {
                            return false;
                        }
                    }
                    for punct in STANDALONE_PUNCTUATION {
                        if s.trim().starts_with(*punct) {
                            return false;
                        }
                    }
                    if lower.contains(INCOMPLETE_QUOTE_PATTERN)
                        && !lower.contains(INCOMPLETE_QUOTE_EXCEPTION)
                    {
                        return false;
                    }
                    for month in MONTH_PATTERNS {
                        if lower.starts_with(month) {
                            return false;
                        }
                    }
                    for pattern in INCOMPLETE_PARENTHETICAL_PATTERNS {
                        if lower.contains(pattern) {
                            return false;
                        }
                    }
                    for ending in SENTENCE_END_INCOMPLETE {
                        if s.ends_with(ending) {
                            return false;
                        }
                    }
                    true
                })
                .take(MAX_DEEP_PARAGRAPHS_TO_SHOW)
                .collect();

            let answer = if think {
                if let Some(ia) = ia_answer {
                    let prompt_lower = prompt.to_ascii_lowercase();
                    let prompt_asks_history = HISTORY_KEYWORDS
                        .iter()
                        .any(|&keyword| prompt_lower.contains(keyword));

                    let filtered_ia = if !prompt_asks_history {
                        let split_result = split_sentences_with_punct(&ia);
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
                            ia.clone()
                        } else {
                            filtered.join(" ")
                        }
                    } else {
                        ia.clone()
                    };

                    let prefixed_ia = if ADVANTAGE_KEYWORDS
                        .iter()
                        .any(|&keyword| prompt_lower.contains(keyword))
                    {
                        let topic_name = extract_topic(&prompt);
                        format!(
                            "The advantages of {} are as follows. {}",
                            topic_name, filtered_ia
                        )
                    } else {
                        filtered_ia
                    };

                    if filtered_deep.len() >= MIN_FILTERED_PARAGRAPHS {
                        let deep_content = filtered_deep.join("\n\n");
                        Some(format!("{}\n\n{}", prefixed_ia, deep_content))
                    } else {
                        Some(prefixed_ia)
                    }
                } else {
                    if filtered_deep.len() >= MIN_FILTERED_PARAGRAPHS {
                        Some(filtered_deep.join("\n\n"))
                    } else {
                        index.answer(&topic, top_k)
                    }
                }
            } else {
                if let Some(ia) = ia_answer {
                    if filtered_deep.len() >= MIN_FILTERED_PARAGRAPHS {
                        let deep_content = filtered_deep.join("\n\n");
                        Some(format!("{}\n\n{}", ia, deep_content))
                    } else {
                        info!(
                            "  🔍 Deep mode: insufficient quality paragraphs ({}), using IA only",
                            filtered_deep.len()
                        );
                        Some(ia)
                    }
                } else {
                    if filtered_deep.len() >= MIN_FILTERED_PARAGRAPHS {
                        let deep_content = filtered_deep.join("\n\n");
                        Some(deep_content)
                    } else {
                        info!(
                            "  🔍 Deep mode: insufficient quality paragraphs, falling back to regular query"
                        );
                        index
                            .answer(&eval_queries[0], top_k)
                            .or_else(|| index.answer(&topic, top_k))
                    }
                }
            };
            (chunks, answer)
        } else {
            let answer = index
                .answer(&eval_queries[0], top_k)
                .or_else(|| index.answer(&topic, top_k));
            (chunks, answer)
        }
    } else {
        let chunks = index
            .query(&kb_query, top_k)
            .into_iter()
            .map(|c| c.text.clone())
            .collect();
        let answer = index
            .answer(&kb_query, top_k)
            .or_else(|| index.answer(&topic, top_k));
        (chunks, answer)
    };

    let (answer_text, is_symbolic_fallback) = match combined_answer {
        Some(answer) if is_quality_answer(&answer) => {
            info!("  ✅ KB answer found ({} chars)", answer.len());

            if think && !deep {
                let prompt_lower = prompt.to_ascii_lowercase();
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
                        answer
                    } else {
                        filtered.join(" ")
                    }
                } else {
                    answer
                };

                (assemble_paragraphs_for(&filtered_answer, run_seed, &kb_query), false)
            } else {
                (answer, false)
            }
        }
        Some(answer) => {
            info!(
                "  🔀 KB answer too sparse ({} chars). Building symbolic response",
                answer.len()
            );
            (build_symbolic_response(&prompt, &cold_corpus), true)
        }
        None => {
            info!("  🔀 No KB answer. Building symbolic response");
            (build_symbolic_response(&prompt, &cold_corpus), true)
        }
    };

    divider("Expert Inference");

    info!(
        "  ⚙️  Config      : depth={} CF≥{:.2}",
        inference_depth, cf_threshold
    );

    let chunk_refs: Vec<&str> = kb_chunks.iter().map(|s| s.as_str()).collect();
    let rule_base = RuleBase::from_chunks(&chunk_refs);
    let expert_result = if rule_base.rules.len() >= DEFAULT_MIN_RULE_COUNT {
        let engine = InferenceEngine::new(inference_depth, cf_threshold);
        engine.reason(&prompt, &rule_base)
    } else {
        None
    };

    let (answer_text, is_symbolic_fallback) = if is_symbolic_fallback {
        (answer_text, true)
    } else if deep {
        (answer_text, false)
    } else if let Some(ref result) = expert_result {
        info!(
            "  🧠 Inference: conclusion={:?} CF={:.2} depth={} rules={}",
            result.conclusion,
            result.certainty,
            result.depth,
            rule_base.rules.len()
        );
        info!("  📋 Explanation: {} steps", result.trace.len());
        if verbose && !result.trace.is_empty() {
            info!("  💬 {}", result.trace.to_prose());
        }
        let builder = ProseBuilder::new(2, 5);
        let expert_prose = builder.build(result, &chunk_refs, run_seed);
        if expert_prose.split_whitespace().count() >= crate::constants::app::MIN_EXPERT_PROSE_WORDS
        {
            let deduplicator = GlobalDeduplicator::new(0.65);
            let merged = format!("{}\n\n{}", answer_text, expert_prose);
            (deduplicator.deduplicate_text(&merged), false)
        } else {
            (answer_text, false)
        }
    } else {
        info!(
            "  ℹ  Expert inference: no conclusion (rules={}, threshold={:.2})",
            rule_base.rules.len(),
            cf_threshold
        );
        (answer_text, false)
    };

    divider("Stochastic Enhancement");
    let effective_prob = match probability {
        Some(p) => p.clamp(PROBABILITY_MIN, PROBABILITY_MAX),
        None => DEFAULT_STOCHASTIC_PROBABILITY,
    };
    let enriched_answer = if is_symbolic_fallback || answer_text.is_empty() || no_frame || think {
        answer_text.clone()
    } else {
        PhrasesMixer::global().apply_opener_if_needed(&answer_text, run_seed)
    };
    let enhanced = if is_symbolic_fallback {
        StochasticEnhancer::new(0.0).enhance_detailed(&enriched_answer)
    } else {
        StochasticEnhancer::new(effective_prob).enhance_detailed(&enriched_answer)
    };
    info!(
        "  🎲 substitutions={} language={:?} prob={:.2}",
        enhanced.substitutions, enhanced.language, effective_prob
    );

    let final_text = if deep || no_frame {
        enhanced.text.clone()
    } else if think {
        if TaskDecomposer::is_creative_task(&prompt) {
            let lang = TaskDecomposer::extract_code_language(&prompt)
                .unwrap_or_else(|| DEFAULT_CODE_LANGUAGE.to_string());
            let kb_corpus: String = kb_chunks.join(" ");
            let task_desc = topic.clone();
            let code = CodeResponder::generate(&lang, &task_desc, &kb_corpus);
            let code_block = CodeResponder::format_code_block(&lang, &code);
            let frame =
                ResponseFramer::opening_phrase(prompt_mode, &topic, &prompt, &enhanced.text);
            if frame.is_empty() {
                format!("{}\n\n{}", enhanced.text.trim(), code_block)
            } else {
                format!("{}\n\n{}", frame.trim_end(), code_block)
            }
        } else {
            let framing_topic = derive_framing_topic(&prompt, prompt_mode, &topic);
            let framed =
                ResponseFramer::frame(prompt_mode, &framing_topic, &prompt, &enhanced.text);
            if matches!(prompt_mode, ReasoningMode::Analogical) {
                restructure_analogical_response(&framed, &topic, run_seed)
            } else {
                framed
            }
        }
    } else {
        enhanced.text.clone()
    };

    let answer_text = if deep && !web_result.all_scraped_urls.is_empty() {
        let mut out = final_text.clone();
        out.push_str(DEEP_SEARCH_URL_FOOTER_HEADER);
        for url in &web_result.all_scraped_urls {
            out.push_str(DEEP_SEARCH_URL_FOOTER_BULLET);
            out.push_str(url);
            out.push('\n');
        }
        out
    } else {
        final_text
    };

    if verbose {
        divider("TRAM Response");
        info!("  {}", answer_text);
    } else {
        println!("{}", answer_text);
    }

    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
