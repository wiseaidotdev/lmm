// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Application Entry-Point
//!
//! This module wires the `crate::cli::commands` argument parser to the
//! individual subsystem functions (encode, simulate, discover, predict,
//! generate text, render images, TRAM reasoning, etc.).
//!
//! All public functionality is gated behind the `cli` Cargo feature so the
//! library can be used as a pure Rust or Python/Node.js dependency without
//! pulling in the CLI stack.
//!
//! ## TRAM Default
//!
//! When the user invokes `lmm -p 'question'` without a subcommand, the
//! `command` field of [`crate::cli::commands::Cli`] is `None` and the
//! `prompt` field is `Some(question)`.  In that case `run_cli_entry` falls
//! through to [`handle_tram`], which is identical to `lmm tram -p 'question'`.

#[cfg(feature = "cli")]
use {
    crate::{
        causal::CausalGraph,
        cli::commands::{
            Cli,
            Commands::{
                Causal, Consciousness as CmdConsciousness, Decode, Discover, Encode, Essay,
                Field as CmdField, Imagen, Paragraph, Physics, Predict, Sentence, Simulate,
                Summarize, Tram,
            },
        },
        cognition::{
            knowledge::{KnowledgeIndex, KnowledgeSource, ingest as cognition_ingest},
            r#loop::ThinkLoop,
            search::SearchOracle,
            thought::AdaptiveThoughtEngine,
        },
        consciousness::Consciousness,
        discovery::SymbolicRegression,
        encode::{decode_from_parts, decode_message, encode_text},
        error::{LmmError, Result as LmmResult},
        expert::{inference::InferenceEngine, rule::RuleBase},
        field::Field,
        generation::{
            dedup::GlobalDeduplicator, filter::SentenceFilter, prose::ProseBuilder,
            shaper::ParagraphShaper,
        },
        imagen::{ImagenParams, StyleMode, render as imagen_render},
        lexicon::Lexicon,
        physics::{HarmonicOscillator, LorenzSystem, Pendulum, SIRModel},
        predict::TextPredictor,
        simulation::Simulator,
        stochastic::StochasticEnhancer,
        tensor::Tensor,
        text::{EssayGenerator, ParagraphGenerator, SentenceGenerator, TextSummarizer},
        traits::{Causal as CausalTrait, Simulatable},
    },
    clap::Parser,
    std::{
        fs, io,
        path::Path,
        process::exit,
        time::{SystemTime, UNIX_EPOCH},
    },
    tracing::{Event, Subscriber, error, info},
    tracing_appender::rolling,
    tracing_subscriber::{
        Layer, Registry, filter,
        fmt::{self as ts_fmt, FmtContext, FormatEvent, FormatFields},
        prelude::__tracing_subscriber_SubscriberExt,
        registry::LookupSpan,
    },
};

#[cfg(all(feature = "cli", feature = "net"))]
use {
    crate::{
        cli::commands::Commands::Ask,
        net::{
            SearchAggregator, corpus_from_response, corpus_from_results, corpus_from_results_raw,
        },
    },
    duckduckgo::{browser::Browser, user_agents::get as get_ua},
};

#[cfg(feature = "cli")]
struct NoLevelFormatter;

#[cfg(feature = "cli")]
impl<S, N> FormatEvent<S, N> for NoLevelFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: ts_fmt::format::Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        ctx.format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

#[cfg(feature = "cli")]
fn setup_logging(verbose: bool) -> anyhow::Result<()> {
    let console_level = if verbose {
        filter::LevelFilter::INFO
    } else {
        filter::LevelFilter::WARN
    };
    let file_appender = rolling::daily("logs", "lmm_log");

    let console_layer = ts_fmt::Layer::new()
        .compact()
        .without_time()
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_target(false)
        .with_writer(io::stdout)
        .event_format(NoLevelFormatter)
        .with_filter(console_level);

    let file_layer = ts_fmt::Layer::new()
        .compact()
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_target(true)
        .with_writer(file_appender)
        .with_filter(filter::LevelFilter::DEBUG);

    let subscriber = Registry::default().with(console_layer).with(file_layer);
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

#[cfg(feature = "cli")]
fn load_source(input: &str, text: String) -> LmmResult<String> {
    if input == "-" {
        Ok(text)
    } else {
        fs::read_to_string(input)
            .map_err(|e| LmmError::Perception(e.to_string()))
            .map(|s| s.trim_end().to_string())
    }
}

#[cfg(feature = "cli")]
fn char_display_width(cp: u32) -> usize {
    match cp {
        0xFE00..=0xFE0F | 0xE0100..=0xE01EF => 0,
        0x200B | 0x200C | 0x200D | 0xFEFF => 0,
        0x1100..=0x115F
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0x3400..=0x9FFF
        | 0xA960..=0xA97F
        | 0xAC00..=0xD7FF
        | 0xF900..=0xFAFF
        | 0xFE10..=0xFE19
        | 0xFE30..=0xFE6F
        | 0xFF01..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x1B000..=0x1B0FF
        | 0x1F004
        | 0x1F0CF
        | 0x1F300..=0x1FFFF
        | 0x20000..=0x2FFFD
        | 0x30000..=0x3FFFD => 2,
        _ => 1,
    }
}

#[cfg(feature = "cli")]
fn display_width(s: &str) -> usize {
    s.chars().map(|c| char_display_width(c as u32)).sum()
}

#[cfg(feature = "cli")]
fn banner(title: &str, icon: &str) {
    let width = 54usize;
    let label = format!("  {}  {}", icon, title);
    let pad = width.saturating_sub(display_width(&label));
    info!("");
    info!("╔{}╗", "═".repeat(width));
    info!("║{}{}║", label, " ".repeat(pad));
    info!("╚{}╝", "═".repeat(width));
    info!("");
}

#[cfg(feature = "cli")]
fn divider(label: &str) {
    let dashes = "-".repeat(46usize.saturating_sub(label.chars().count() + 4));
    info!("");
    info!("-- {} {}", label, dashes);
}

#[cfg(feature = "cli")]
fn maybe_enhance(text: &str, stochastic: bool, probability: f64) -> String {
    if stochastic {
        StochasticEnhancer::new(probability).enhance(text)
    } else {
        text.to_string()
    }
}

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
) -> anyhow::Result<()> {
    banner("TRAM · Text Reasoning and Applied Mathematics", "🧮");
    info!("  ❓ Prompt    : {:?}", prompt);
    info!("  🔁 Iterations: {}", iterations);
    info!(
        "  🔍 Inference : depth={} CF≥{:.2}",
        inference_depth, cf_threshold
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

    prefetch_web(&prompt, &mut index, kb_provided, verbose).await;

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

    let convergence_threshold = 0.60;
    let think_iterations = if index.is_empty() {
        iterations
    } else {
        iterations.min(4)
    };
    let mut lp = ThinkLoop::builder(ate_goal)
        .max_iterations(think_iterations)
        .convergence_threshold(convergence_threshold)
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

    let top_k = 6 + (run_seed % 5) as usize;
    let kb_chunks: Vec<String> = index
        .query(&prompt, top_k)
        .into_iter()
        .map(|c| c.text.clone())
        .collect();

    let (answer_text, from_kb) = match index.answer(&prompt, top_k) {
        Some(answer) if is_quality_answer(&answer) => {
            info!("  ✅ KB answer found ({} chars)", answer.len());
            let prose = assemble_paragraphs_for(&answer, run_seed, &prompt);
            (prose, true)
        }
        Some(_) => {
            info!("  🔀 KB answer too sparse. Building symbolic response");
            let text = build_symbolic_response(&prompt, &cold_corpus);
            (text, false)
        }
        None => {
            info!("  🔀 No KB answer. Building symbolic response");
            let text = build_symbolic_response(&prompt, &cold_corpus);
            (text, false)
        }
    };

    divider("Expert Inference");

    info!(
        "  ⚙️  Config      : depth={} CF≥{:.2}",
        inference_depth, cf_threshold
    );

    let chunk_refs: Vec<&str> = kb_chunks.iter().map(|s| s.as_str()).collect();
    let rule_base = RuleBase::from_chunks(&chunk_refs);
    let engine = InferenceEngine::new(inference_depth, cf_threshold);
    let expert_result = engine.reason(&prompt, &rule_base);

    let answer_text = if let Some(ref result) = expert_result {
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
        if expert_prose.split_whitespace().count() >= 30 {
            let deduplicator = GlobalDeduplicator::new(0.65);
            let merged = format!("{}\n\n{}", answer_text, expert_prose);
            let deduped = deduplicator.deduplicate_text(&merged);
            assemble_paragraphs_for(&deduped, run_seed, &prompt)
        } else {
            answer_text
        }
    } else {
        info!(
            "  ℹ️  Expert inference: no conclusion (rules={}, threshold={:.2})",
            rule_base.rules.len(),
            cf_threshold
        );
        answer_text
    };

    divider("Stochastic Enhancement");
    let effective_prob = match probability {
        Some(p) => p.clamp(0.0, 1.0),
        None => {
            if from_kb {
                0.05
            } else {
                0.10
            }
        }
    };
    let enhanced = StochasticEnhancer::new(effective_prob)
        .curated_only()
        .enhance_detailed(&answer_text);
    info!(
        "  🎲 substitutions={} language={:?} prob={:.2}",
        enhanced.substitutions, enhanced.language, effective_prob
    );

    if verbose {
        divider("TRAM Response");
        info!("  {}", enhanced.text);
    } else {
        println!("{}", enhanced.text);
    }

    Ok(())
}

/// Returns `true` when `answer` contains enough substance to present to the user.
///
/// A "quality" answer has an average of ≥ 8 words per sentence. Short sentences
/// indicate the index was populated with page-title strings rather than real
/// prose (a common outcome when DuckDuckGo Lite returns no snippets).
///
/// # Arguments
///
/// * `answer` - Raw text from `KnowledgeIndex::answer`.
///
/// # Time Complexity
///
/// O(|answer|)
#[cfg(feature = "cli")]
fn is_quality_answer(answer: &str) -> bool {
    let word_count = answer.split_whitespace().count();
    if word_count < 15 {
        return false;
    }
    let sentences: Vec<&str> = answer
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| {
            !s.is_empty()
                && s.split_whitespace().count() >= 5
                && !s.chars().next().is_some_and(|c| c.is_ascii_digit())
        })
        .collect();
    let sentence_count = sentences.len().max(1);
    let avg_words_per_sentence = word_count / sentence_count;

    if avg_words_per_sentence < 6 {
        return false;
    }

    let verb_count = sentences
        .iter()
        .filter(|s| {
            let lower = s.to_lowercase();
            lower.contains(" is ")
                || lower.contains(" are ")
                || lower.contains(" was ")
                || lower.contains(" were ")
                || lower.contains(" has ")
                || lower.contains(" have ")
                || lower.contains(" provides ")
                || lower.contains(" enables ")
                || lower.contains(" uses ")
                || lower.contains(" includes ")
                || lower.contains(" supports ")
                || lower.contains(" allows ")
                || lower.contains(" makes ")
                || lower.contains(" creates ")
                || lower.contains(" consists ")
                || lower.contains(" contains ")
                || lower.contains(" designed ")
                || lower.contains(" built ")
                || lower.contains(" formed ")
                || lower.contains(" released ")
                || lower.contains(" developed ")
                || lower.contains(" began ")
                || lower.contains(" defines ")
                || lower.contains(" describe ")
        })
        .count();
    verb_count * 10 >= sentence_count * 3
}

/// Assembles and shapes final output paragraphs, optionally filtering sentences
/// that open with an off-topic proper noun not related to `topic`.
#[cfg(feature = "cli")]
fn assemble_paragraphs_for(text: &str, run_seed: u64, topic: &str) -> String {
    let topic_cleaned = extract_topic(topic);
    let topic_ref = if topic_cleaned.is_empty() {
        topic
    } else {
        topic_cleaned.as_str()
    };
    let topic_word: String = topic_ref
        .split_whitespace()
        .find(|w| {
            w.len() > 3
                && !matches!(
                    *w,
                    "what"
                        | "does"
                        | "how"
                        | "the"
                        | "and"
                        | "for"
                        | "with"
                        | "about"
                        | "install"
                        | "make"
                        | "build"
                        | "create"
                        | "setup"
                        | "configure"
                        | "learn"
                        | "understand"
                        | "explain"
                        | "describe"
                        | "find"
                        | "show"
                        | "tell"
                        | "give"
                        | "help"
                        | "work"
                        | "works"
                        | "working"
                        | "used"
                        | "using"
                        | "uses"
                        | "need"
                        | "needs"
                        | "want"
                        | "that"
                        | "this"
                        | "into"
                        | "from"
                        | "have"
                        | "been"
                        | "will"
                        | "would"
                        | "could"
                        | "should"
                        | "when"
                        | "where"
                        | "which"
                        | "who"
                        | "why"
                        | "also"
                )
        })
        .unwrap_or("")
        .to_lowercase();

    let mut sentences: Vec<String> = split_sentences_with_punct(text)
        .into_iter()
        .map(strip_discourse_connector)
        .filter(|s| SentenceFilter::is_acceptable(s))
        .filter(|s| {
            static DANGLING_PREPOSITIONS: &[&str] = &[
                "in", "at", "of", "to", "for", "on", "by", "with", "from", "into", "onto", "upon",
                "about", "over", "under", "between", "among", "through", "within", "without",
                "during", "and", "or", "but", "the", "a", "an",
            ];
            let last = s
                .trim_end_matches(['.', '!', '?'])
                .split_whitespace()
                .last()
                .unwrap_or("")
                .to_ascii_lowercase();
            !DANGLING_PREPOSITIONS.contains(&last.as_str())
        })
        .filter(|s| {
            if topic_word.is_empty() {
                return true;
            }
            let first = s.split_whitespace().next().unwrap_or("");
            if first.len() < 4 {
                return true;
            }
            let mut chars = first.chars();
            let starts_upper = chars.next().map(|c| c.is_uppercase()).unwrap_or(false);
            let rest_lower = chars.all(|c| c.is_lowercase());
            if !starts_upper || !rest_lower {
                return true;
            }
            let first_lower = first.to_lowercase();
            if first_lower.contains(&topic_word) || topic_word.contains(&first_lower) {
                return true;
            }
            if s.to_lowercase().contains(&topic_word) {
                return true;
            }
            false
        })
        .collect();

    if sentences.is_empty() {
        return text.to_string();
    }

    let n = sentences.len();
    let mut seed = run_seed;
    for i in (1..n).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        sentences.swap(i, j);
    }

    let deduplicator = GlobalDeduplicator::new(0.65);
    let mut sentences = deduplicator.deduplicate(sentences);

    if sentences.is_empty() {
        return text.to_string();
    }

    if let Some(good_pos) = sentences
        .iter()
        .position(|s| SentenceFilter::is_good_opener(s))
    {
        sentences.swap(0, good_pos);
    }

    let shaper = ParagraphShaper::new(2, 2, 5, 40, 180);
    shaper.shape(&sentences, run_seed)
}

/// Strips a leading discourse connector phrase from a sentence.
///
/// Connectors like `"Furthermore, "`, `"In addition, "`, `"As a result, "` are
/// prepended by [`ProseBuilder`] and [`ParagraphShaper`] when building
/// non-first paragraphs.  When sentences are later re-shuffled through
/// [`assemble_paragraphs`], these stale connectors cause double-stacking
/// (e.g. `"To elaborate, Furthermore, rust supports..."`).  This function
/// removes them so the shaper can apply fresh connectors at the right positions.
///
/// # Time Complexity
///
/// O(|sentence|)
#[cfg(feature = "cli")]
fn strip_discourse_connector(sentence: String) -> String {
    static CONNECTOR_PREFIXES: &[&str] = &[
        "Furthermore, ",
        "In addition, ",
        "Notably, ",
        "Moreover, ",
        "As a result, ",
        "In practice, ",
        "Interestingly, ",
        "Consequently, ",
        "Building on this, ",
        "To elaborate, ",
        "More specifically, ",
        "On the technical side, ",
        "From a practical standpoint, ",
        "In this context, ",
        "Also, ",
        "However, ",
        "Therefore, ",
        "Thus, ",
        "Hence, ",
        "More precisely, ",
        "This is because ",
        "To be specific, ",
        "It is well established that ",
        "At its core, ",
        "Put simply, ",
        "Broadly speaking, ",
        "In technical terms, ",
        "By design, ",
        "In essence, ",
        "Based on observed knowledge ",
        "To elaborate on this, ",
    ];
    for prefix in CONNECTOR_PREFIXES {
        if let Some(rest) = sentence.strip_prefix(prefix) {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                let mut chars = trimmed.chars();
                let first_up: String = chars
                    .next()
                    .map(|c| c.to_uppercase().collect::<String>())
                    .unwrap_or_default();
                return first_up + chars.as_str();
            }
        }
    }
    sentence
}

/// Splits text into complete sentences with terminal punctuation attached.
///
/// Sentences below 8 words are silently discarded.
///
/// Handles common English abbreviations (`e.g.`, `i.e.`, `vs.`, `Dr.`,
/// `Mr.`, `Mrs.`, `No.`, `Fig.`, `cf.`) to prevent spurious sentence splits
/// at their trailing periods.
///
/// # Time Complexity
///
/// O(|text|)
#[cfg(feature = "cli")]
fn split_sentences_with_punct(text: &str) -> Vec<String> {
    static ABBREVIATIONS: phf::Set<&'static str> = phf::phf_set! {
        "e.g", "i.e", "vs", "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
        "no", "fig", "cf", "etc", "approx", "dept", "est", "govt", "inc",
        "ltd", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec",
    };

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();

    for (i, &ch) in chars.iter().enumerate() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let next_char = chars.get(i + 1).copied();
            let next_lower = next_char.map(|c| c.is_ascii_lowercase()).unwrap_or(false);
            let next_space = next_char.map(|c| c == ' ').unwrap_or(false);
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = next_char.map(|c| c.is_ascii_digit()).unwrap_or(false);

            if ch == '.' && (next_lower || (prev_digit && next_digit)) {
                continue;
            }

            if ch == '.' && next_space {
                let word_before: String = cur
                    .trim_end_matches('.')
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .chars()
                    .filter(|c| c.is_ascii_alphabetic() || *c == '.')
                    .collect::<String>()
                    .to_ascii_lowercase();
                let stripped = word_before.trim_end_matches('.');
                if ABBREVIATIONS.contains(stripped) {
                    continue;
                }
            }

            let next_upper = next_char.map(|c| c.is_ascii_uppercase()).unwrap_or(false);
            if ch == '.'
                && !next_upper
                && !next_char.map(|c| c == '\n').unwrap_or(false)
                && next_char.is_some()
                && !next_space
            {
                continue;
            }

            let s = cur.trim().to_string();
            if s.split_whitespace().count() >= 8 {
                out.push(s);
            }
            cur.clear();
        }
    }

    let tail = cur.trim().to_string();
    if tail.split_whitespace().count() >= 8 {
        if !tail.ends_with('.') && !tail.ends_with('!') && !tail.ends_with('?') {
            out.push(format!("{}.", tail));
        } else {
            out.push(tail);
        }
    }

    let _ = len;
    out
}

/// Builds a topic-aware symbolic response when no KB answer is available.
///
/// First attempts to summarise the ThinkLoop cold-memory corpus. If that
/// corpus is too thin, returns a graceful "no information available" message
/// grounded in the topic extracted from the prompt, rather than producing
/// mathematical word-salad from `ParagraphGenerator`.
///
/// # Arguments
///
/// * `prompt`       - Original user prompt (used as topic seed).
/// * `cold_corpus`  - Concatenated ThinkLoop cold-store content (may be empty).
///
/// # Time Complexity
///
/// O(|cold_corpus| + |prompt|)
#[cfg(feature = "cli")]
fn build_symbolic_response(prompt: &str, cold_corpus: &str) -> String {
    if cold_corpus.split_whitespace().count() >= 20 {
        let summarizer = TextSummarizer::new(5, 8, 3);
        if let Ok(sentences) = summarizer.summarize_with_query(cold_corpus, prompt) {
            let joined = sentences.join(" ");
            if joined.split_whitespace().count() >= 10 {
                return joined;
            }
        }
    }
    let topic = extract_topic(prompt);
    let topic_display = if topic.is_empty() {
        prompt
    } else {
        topic.as_str()
    };
    format!(
        "No detailed information was found for \"{topic_display}\". \
         This may be because the topic is very specific, the web query returned no results, \
         or the knowledge base does not yet contain content on this subject. \
         Try rephrasing your query or adding more context."
    )
}

/// Extracts a clean topic keyword phrase from a natural-language prompt by
/// stripping common question prefixes and filler words.
///
/// The DuckDuckGo Instant Answer API returns rich Wikipedia summaries for
/// concise noun phrases ("Rust programming language") but nothing for
/// question forms ("what is rust programming"). This function normalises the
/// prompt before the IA query while keeping the full text for ThinkLoop.
///
/// # Time Complexity
///
/// O(|prompt|)
#[cfg(feature = "cli")]
fn extract_topic(prompt: &str) -> String {
    static FILLER_PREFIXES: &[&str] = &[
        "what is a ",
        "what is an ",
        "what is the ",
        "what is ",
        "what are the ",
        "what are ",
        "what was ",
        "what were ",
        "how does a ",
        "how does an ",
        "how does ",
        "how do ",
        "how is ",
        "how are ",
        "how to ",
        "explain ",
        "describe ",
        "define ",
        "tell me about ",
        "what's ",
        "whats ",
        "who is ",
        "who was ",
        "why is ",
        "why are ",
        "why does ",
        "why do ",
        "where is ",
        "when is ",
        "give me information about ",
        "give me info on ",
    ];
    let lower = prompt.to_lowercase();
    for prefix in FILLER_PREFIXES {
        if lower.starts_with(prefix) {
            let raw = prompt[prefix.len()..]
                .trim()
                .trim_end_matches('?')
                .trim_end_matches('.')
                .trim_end_matches('!')
                .trim();
            let stripped = strip_leading_articles(raw);
            return strip_trailing_verb(trim_predicate(&stripped));
        }
    }
    let clean = prompt
        .trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim_end_matches('!')
        .trim();
    strip_leading_articles(clean)
}

/// Strips a leading article ("a", "an", "the") from a noun phrase.
///
/// # Time Complexity
///
/// O(|s|)
#[cfg(feature = "cli")]
fn strip_leading_articles(s: &str) -> String {
    let lower = s.to_lowercase();
    for article in &["a ", "an ", "the "] {
        if lower.starts_with(article) {
            return s[article.len()..].trim().to_string();
        }
    }
    s.to_string()
}

/// Trims a topic string at the first predicate word, keeping only the subject noun phrase.
///
/// When a question like `"why does Rust not have a garbage collector"` is stripped
/// of its question prefix, the remainder `"Rust not have a garbage collector"` still
/// contains the predicate.  This function truncates at the first auxiliary, negation,
/// or main-verb word so only `"Rust"` is returned, a far better search term.
///
/// Only applies when the input has 3 or more words to avoid over-truncating short topics.
///
/// # Time Complexity
///
/// O(|s|)
#[cfg(feature = "cli")]
fn trim_predicate(s: &str) -> String {
    static PREDICATE_WORDS: &[&str] = &[
        "not", "no", "never", "have", "has", "had", "is", "are", "was", "were", "do", "does",
        "did", "will", "would", "could", "should", "can", "cannot", "work", "works", "make",
        "makes", "use", "uses", "need", "needs", "allow", "allows", "prevent", "prevents",
        "require", "requires", "cause", "causes",
    ];
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < 3 {
        return s.to_string();
    }
    let cut = words
        .iter()
        .skip(1)
        .position(|w| PREDICATE_WORDS.contains(&w.to_ascii_lowercase().as_str()))
        .map(|i| i + 1)
        .unwrap_or(words.len());
    words[..cut].join(" ")
}

/// Strips a trailing common verb from a topic string extracted from a question.
///
/// For example, `"how does tokio work"` → `extract_topic` strips `"how does "`
/// yielding `"tokio work"`. This function further strips `" work"` to give
/// just `"tokio"`, which produces cleaner DuckDuckGo Instant Answer hits.
///
/// # Time Complexity
///
/// O(|s|)
#[cfg(feature = "cli")]
fn strip_trailing_verb(s: String) -> String {
    static TRAILING_VERBS: &[&str] = &[
        " work",
        " works",
        " function",
        " functions",
        " operate",
        " operates",
        " run",
        " runs",
        " do",
        " does",
        " mean",
        " means",
        " help",
        " helps",
        " differ",
        " differs",
    ];
    let lower = s.to_lowercase();
    for verb in TRAILING_VERBS {
        if lower.ends_with(verb) {
            return s[..s.len() - verb.len()].trim().to_string();
        }
    }
    s
}

/// Issues a DuckDuckGo pre-fetch when no KB is provided and the `net` feature is active.
///
/// Results are ingested directly into `index` before the ThinkLoop runs,
/// giving the IDF retriever real web content to rank against.
///
/// This function is a no-op when compiled without the `net` feature.
///
/// # Arguments
///
/// * `prompt`      - The search query to issue.
/// * `index`       - The `KnowledgeIndex` to populate.
/// * `kb_provided` - When `true`, a local KB was supplied and the web fetch is skipped.
/// * `verbose`     - When `true`, diagnostic lines are printed.
#[cfg(feature = "cli")]
async fn prefetch_web(prompt: &str, index: &mut KnowledgeIndex, kb_provided: bool, verbose: bool) {
    prefetch_web_inner(prompt, index, kb_provided, verbose).await;
}

#[cfg(all(feature = "cli", feature = "net"))]
async fn prefetch_web_inner(
    prompt: &str,
    index: &mut KnowledgeIndex,
    kb_provided: bool,
    verbose: bool,
) {
    if kb_provided {
        return;
    }

    divider("Web Knowledge");
    let browser = Browser::new();
    let ua = get_ua("firefox").unwrap_or("Mozilla/5.0");
    let mut ingested = 0usize;
    let mut prose_chunks = 0usize;

    let topic_query = extract_topic(prompt);
    let topic_lang = format!("{} programming language", topic_query);
    let topic_lang2 = format!("{} language", topic_query);
    let topic_what = format!("what is {}", topic_query);
    let topic_library = format!("{} library", topic_query);
    let topic_framework = format!("{} framework", topic_query);

    let topic_words: Vec<String> = topic_query
        .split_whitespace()
        .filter(|w| w.len() >= 3)
        .map(|w| format!("{} programming language", w))
        .collect();

    let mut ia_candidates: Vec<&str> = vec![
        topic_query.as_str(),
        topic_lang.as_str(),
        topic_lang2.as_str(),
        topic_library.as_str(),
        topic_framework.as_str(),
        topic_what.as_str(),
        prompt,
    ];
    for tw in &topic_words {
        if !ia_candidates.contains(&tw.as_str()) {
            ia_candidates.push(tw.as_str());
        }
    }

    let mut ia_winner: Option<&str> = None;
    for query in ia_candidates.iter().copied() {
        if prose_chunks > 0 {
            break;
        }
        let encoded = query
            .chars()
            .map(|c| if c == ' ' { '+' } else { c })
            .collect::<String>();
        let ia_path = format!("?q={}&ia=web", encoded);
        if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
            if let Some(text) = resp.abstract_text.filter(|s| !s.trim().is_empty()) {
                let wc = text.split_whitespace().count();
                let added = index.ingest_text("web-ia", text.trim());
                if added > 0 && wc >= 20 {
                    prose_chunks += added;
                }
                ingested += added;
            }
            if let Some(def) = resp.definition.filter(|s| !s.trim().is_empty()) {
                let wc = def.split_whitespace().count();
                let added = index.ingest_text("web-ia", def.trim());
                if added > 0 && wc >= 20 {
                    prose_chunks += added;
                }
                ingested += added;
            }
            if let Some(ans) = resp.answer.filter(|s| !s.trim().is_empty()) {
                let wc = ans.split_whitespace().count();
                let added = index.ingest_text("web-ia", ans.trim());
                if added > 0 && wc >= 20 {
                    prose_chunks += added;
                }
                ingested += added;
            }
            for topic in resp.related_topics.iter().take(10) {
                if let Some(text) = topic.text.as_deref().filter(|s| !s.trim().is_empty()) {
                    let wc = text.split_whitespace().count();
                    if wc >= 20 && SentenceFilter::is_acceptable(text.trim()) {
                        let added = index.ingest_text("web-ia", text.trim());
                        prose_chunks += added;
                        ingested += added;
                    } else if wc >= 8 && SentenceFilter::is_acceptable(text.trim()) {
                        ingested += index.ingest_text("web-ia", text.trim());
                    }
                }
            }
            if ingested > 0 && ia_winner.is_none() {
                ia_winner = Some(query);
            }
        }
    }

    if prose_chunks > 0 {
        info!(
            "  🌐 Instant Answer [{:?}]: {} chunk(s) ingested ({} prose)",
            ia_winner.unwrap_or(topic_query.as_str()),
            ingested,
            prose_chunks
        );
        return;
    }

    if ingested > 0 {
        info!(
            "  🌐 Instant Answer [{:?}]: {} fragment(s), augmenting with Lite search",
            ia_winner.unwrap_or(topic_query.as_str()),
            ingested
        );
    }

    let mut lite_snippets_found = 0usize;
    let mut lite_q_list: Vec<String> = vec![prompt.to_string()];
    if topic_query.as_str() != prompt {
        lite_q_list.push(topic_query.clone());
    }
    for word in topic_query.split_whitespace() {
        if word.len() >= 3 && !lite_q_list.iter().any(|q| q == word) {
            lite_q_list.push(word.to_string());
        }
    }
    'lite: for lite_q in &lite_q_list {
        match browser
            .lite_search(lite_q.as_str(), "wt-wt", Some(15), ua)
            .await
        {
            Ok(results) if !results.is_empty() => {
                for r in &results {
                    let snippet = r.snippet.trim();
                    let title = r.title.trim();
                    if snippet.split_whitespace().count() >= 8 {
                        let chunk = format!("{} {}", title, snippet);
                        ingested += index.ingest_text("web-lite", chunk.trim());
                        lite_snippets_found += 1;
                    }
                }
                if lite_snippets_found == 0 {
                    info!(
                        "  🌐 Lite search [{:?}]: {} results (titles only - no snippets)",
                        lite_q.as_str(),
                        results.len()
                    );
                } else {
                    info!(
                        "  🌐 Lite search [{:?}]: {} result(s), {} snippet chunk(s) ingested",
                        lite_q.as_str(),
                        results.len(),
                        ingested
                    );
                    break 'lite;
                }
            }
            Ok(_) => {
                if verbose {
                    info!("  🌐 Lite search [{:?}]: no results", lite_q.as_str());
                }
            }
            Err(e) => {
                if verbose {
                    info!("  ⚠️  Lite search [{:?}] error: {e}", lite_q.as_str());
                }
            }
        }
    }
    if lite_snippets_found == 0 && ingested == 0 {
        info!("  🌐 Lite search: no results");
    }
}

#[cfg(feature = "cli")]
pub async fn run_cli_entry(args: Vec<String>) -> anyhow::Result<()> {
    let cli = Cli::parse_from(args);
    setup_logging(cli.verbose)?;

    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            let prompt = cli.prompt.unwrap_or_else(|| {
                eprintln!(
                    "lmm: no subcommand given. Try `lmm --help` or `lmm -p 'your question'`."
                );
                exit(1);
            });
            return handle_tram(
                prompt,
                cli.knowledge_base,
                cli.probability,
                cli.verbose,
                cli.iterations,
                cli.inference_depth,
                cli.cf_threshold,
                cli.think,
                cli.think_steps,
                cli.think_threshold,
                cli.think_patience,
            )
            .await;
        }
    };

    match command {
        Tram {
            prompt,
            knowledge_base,
            probability,
            iterations,
            depth: _,
            inference_depth,
            cf_threshold,
            think,
            think_steps,
            think_threshold,
            think_patience,
        } => {
            handle_tram(
                prompt,
                knowledge_base,
                probability,
                cli.verbose,
                iterations,
                inference_depth,
                cf_threshold,
                think,
                think_steps,
                think_threshold,
                think_patience,
            )
            .await?;
        }

        Simulate { step, steps } => {
            banner("Simulate · Harmonic Oscillator", "🌊");
            let osc = HarmonicOscillator::new(1.0, 1.0, 0.0)?;
            let sim = Simulator { step_size: step };
            let trajectory = sim.simulate_trajectory(&osc, osc.state(), steps)?;
            info!(
                "✅ Simulation complete - {} steps, step_size={}",
                trajectory.len() - 1,
                step
            );
            let final_data = &trajectory.last().unwrap().data;
            if cli.verbose {
                divider("Final State");
                info!("  {:?}", final_data);
            } else {
                println!("{:?}", final_data);
            }
        }

        Discover {
            data_path: _,
            iterations,
        } => {
            banner("Discover · Symbolic Regression", "🔭");
            let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
            let inputs: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
            let sr = SymbolicRegression::new(3, iterations).with_variables(vec!["x".into()]);
            let expr = sr.fit(&inputs, &ys)?;
            if cli.verbose {
                divider("Discovered Equation");
                info!("  f(x) = {}", expr);
            } else {
                println!("{}", expr);
            }
        }

        CmdConsciousness { lookahead } => {
            banner("Consciousness · Perception Loop", "🧠");
            let mut consc = Consciousness::new(Tensor::zeros(vec![4]), lookahead, 0.01);
            let input = vec![128u8, 64, 32, 255];
            let state = consc.tick(&input)?;
            if cli.verbose {
                divider("State");
                info!("  {:?}", state.data);
                divider("Mean Prediction Error");
                info!("  {:.6}", consc.mean_prediction_error());
            } else {
                println!("{:?}\n{:.6}", state.data, consc.mean_prediction_error());
            }
        }

        Physics {
            model,
            steps,
            step_size,
        } => {
            banner("Physics · Dynamical Systems", "⚛️");
            let sim = Simulator { step_size };
            match model.as_str() {
                "lorenz" => {
                    let sys = LorenzSystem::canonical()?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    info!("✅ Lorenz: {} steps", traj.len() - 1);
                    let final_data = &traj.last().unwrap().data;
                    if cli.verbose {
                        divider("Final xyz");
                        info!("  {:?}", final_data);
                    } else {
                        println!("{:?}", final_data);
                    }
                }
                "pendulum" => {
                    let sys = Pendulum::new(9.81, 1.0, 0.3, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    info!("✅ Pendulum: {} steps", traj.len() - 1);
                    let final_data = &traj.last().unwrap().data;
                    if cli.verbose {
                        divider("Final [θ, ω]");
                        info!("  {:?}", final_data);
                    } else {
                        println!("{:?}", final_data);
                    }
                }
                "sir" => {
                    let sys = SIRModel::new(0.3, 0.1, 990.0, 10.0, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    info!("✅ SIR: {} steps", traj.len() - 1);
                    let final_data = &traj.last().unwrap().data;
                    if cli.verbose {
                        divider("Final [S, I, R]");
                        info!("  {:?}", final_data);
                    } else {
                        println!("{:?}", final_data);
                    }
                }
                _ => {
                    let sys = HarmonicOscillator::new(1.0, 1.0, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    info!("✅ Harmonic: {} steps", traj.len() - 1);
                    let s = traj.last().unwrap();
                    let energy = 0.5 * s.data[1] * s.data[1] + 0.5 * s.data[0] * s.data[0];
                    if cli.verbose {
                        divider("Final [x, v]");
                        info!("  {:?}", s.data);
                        divider("Final Energy");
                        info!("  {:.6}", energy);
                    } else {
                        println!("{:?}\n{:.6}", s.data, energy);
                    }
                }
            }
        }

        Causal {
            intervene_node,
            intervene_value,
        } => {
            banner("Causal · Intervention Engine", "🔀");
            let mut graph = CausalGraph::new();
            graph.add_node("x", Some(0.0));
            graph.add_node("y", None);
            graph.add_node("z", None);
            graph.add_edge("x", "y", Some(2.0))?;
            graph.add_edge("y", "z", Some(1.0))?;
            graph
                .nodes
                .iter_mut()
                .find(|n| n.name == "x")
                .unwrap()
                .value = Some(3.0);
            graph.forward_pass()?;
            divider("Before Intervention");
            info!(
                "  x={:?}  y={:?}  z={:?}",
                graph.get_value("x"),
                graph.get_value("y"),
                graph.get_value("z")
            );
            graph.intervene(&intervene_node, intervene_value)?;
            graph.forward_pass()?;
            if cli.verbose {
                divider(&format!("After do({}={})", intervene_node, intervene_value));
                info!(
                    "  x={:?}  y={:?}  z={:?}",
                    graph.get_value("x"),
                    graph.get_value("y"),
                    graph.get_value("z")
                );
            } else {
                println!(
                    "x={:?} y={:?} z={:?}",
                    graph.get_value("x"),
                    graph.get_value("y"),
                    graph.get_value("z")
                );
            }
        }

        CmdField { size, operation } => {
            banner("Field · Differential Operators", "🌐");
            let data: Vec<f64> = (0..size).map(|i| (i as f64).powi(2)).collect();
            let tensor = Tensor::new(vec![size], data)?;
            let field = Field::new(vec![size], tensor)?;
            match operation.as_str() {
                "laplacian" => {
                    let lap = field.compute_laplacian()?;
                    if cli.verbose {
                        divider("∇² f(x)");
                        info!("  {:?}", lap.values.data);
                    } else {
                        println!("{:?}", lap.values.data);
                    }
                }
                _ => {
                    let grad = field.compute_gradient()?;
                    if cli.verbose {
                        divider("∇ f(x)");
                        info!("  {:?}", grad.values.data);
                    } else {
                        println!("{:?}", grad.values.data);
                    }
                }
            }
        }

        Encode {
            input,
            text,
            iterations,
            depth,
        } => {
            banner("Encode · GP Symbolic Compression", "📐");
            let source = load_source(&input, text)?;
            info!("  📝 Input   : {:?}", source);
            info!("  📏 Length  : {} chars", source.len());
            let encoded = encode_text(&source, iterations, depth)?;
            let decoded = decode_message(&encoded)?;
            if cli.verbose {
                divider("Equation");
                info!("  {}", encoded.summary());
                divider("Encoded Data");
                info!("  {}", encoded.to_data_string());
                divider("Round-trip Verify");
                let status = if decoded == source {
                    "✅ PERFECT"
                } else {
                    "⚠️  lossy (residuals correct)"
                };
                info!("  {}", status);
                info!("  Decoded : {:?}", decoded);
                info!("");
                info!("  💾 To decode later:");
                info!(
                    "     lmm decode --equation {:?} --length {} --residuals {:?}",
                    encoded.equation.to_string(),
                    encoded.length,
                    encoded
                        .residuals
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            } else {
                println!(
                    "{} {} {}",
                    encoded.equation,
                    encoded.length,
                    encoded
                        .residuals
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }

        Decode {
            equation,
            length,
            residuals,
        } => {
            banner("Decode · Equation to Text", "🔓");
            info!("  📐 Equation : {}", equation);
            info!("  📏 Length   : {}", length);
            let decoded = decode_from_parts(&equation, length, &residuals)?;
            if cli.verbose {
                divider("Decoded Text");
                info!("  {}", decoded);
            } else {
                println!("{}", decoded);
            }
        }

        Predict {
            input,
            text,
            window,
            predict_length,
            iterations,
            depth,
            dictionary,
            stochastic,
            probability,
        } => {
            banner("Predict · Symbolic Continuation", "🔮");
            let source = load_source(&input, text)?;
            let lexicon = match dictionary {
                Some(path) => Lexicon::load_from(Path::new(&path)).ok(),
                None => Lexicon::load_system().ok(),
            };
            let mut predictor = TextPredictor::new(window, iterations, depth);
            if let Some(lex) = lexicon {
                info!("  📚 Dictionary: {} words loaded", lex.word_count());
                predictor = predictor.with_lexicon(lex);
            }
            let result = predictor.predict_continuation(&source, predict_length)?;
            info!("  📝 Input     : {:?}", source);
            info!("  🪟 Window    : {} words", result.window_used);
            info!("  📈 Trajectory: {}", result.trajectory_equation);
            info!("  🎵 Rhythm    : {}", result.rhythm_equation);
            let cont = maybe_enhance(&result.continuation, stochastic, probability);
            if cli.verbose {
                divider("Continuation");
                info!("  {}{}", source, cont);
            } else {
                println!("{}{}", source, cont);
            }
        }

        Summarize {
            input,
            text,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Summarize · Key Sentence Extraction", "✂️");
            let source = load_source(&input, text)?;
            info!("  📝 Input : {} chars", source.len());
            info!(
                "  📊 Extracting {} key sentences via GP scoring...",
                sentences
            );
            let summarizer = TextSummarizer::new(sentences, iterations, depth);
            match summarizer.summarize(&source) {
                Ok(summary) => {
                    if cli.verbose {
                        divider("Summary");
                    }
                    for (i, sentence) in summary.iter().enumerate() {
                        let enhanced = maybe_enhance(sentence, stochastic, probability);
                        if cli.verbose {
                            info!("  {}. {}", i + 1, enhanced);
                        } else {
                            println!("{}", enhanced);
                        }
                    }
                }
                Err(e) => {
                    error!("  ❌ Summarization failed: {}", e);
                }
            }
        }

        Sentence {
            input,
            text,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Sentence · Single Sentence Generation", "✍️");
            let source = load_source(&input, text)?;
            info!("  🌱 Seed : {:?}", source);
            let sentence_gen = SentenceGenerator::new(iterations, depth);
            match sentence_gen.generate(&source) {
                Ok(sentence) => {
                    let enhanced = maybe_enhance(&sentence, stochastic, probability);
                    if cli.verbose {
                        divider("Generated Sentence");
                        info!("  {}", enhanced);
                    } else {
                        println!("{}", enhanced);
                    }
                }
                Err(e) => {
                    error!("  ❌ Generation failed: {}", e);
                }
            }
        }

        Paragraph {
            input,
            text,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Paragraph · Generate a Paragraph", "📄");
            let source = load_source(&input, text)?;
            info!("  🌱 Seed      : {:?}", source);
            info!("  📊 Sentences : {}", sentences);
            let para_gen = ParagraphGenerator::new(sentences, iterations, depth);
            match para_gen.generate(&source) {
                Ok(paragraph) => {
                    let enhanced = maybe_enhance(&paragraph, stochastic, probability);
                    if cli.verbose {
                        divider("Generated Paragraph");
                        info!("");
                        info!("  {}", enhanced);
                    } else {
                        println!("{}", enhanced);
                    }
                }
                Err(e) => {
                    error!("  ❌ Generation failed: {}", e);
                }
            }
        }

        Essay {
            input,
            text,
            paragraphs,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Essay · Generate a Full Essay", "📖");
            let source = load_source(&input, text)?;
            info!("  🌱 Topic      : {:?}", source);
            info!(
                "  📊 Paragraphs : {} ({} sentences each)",
                paragraphs, sentences
            );
            let essay_gen = EssayGenerator::new(paragraphs, sentences, iterations, depth);
            match essay_gen.generate(&source) {
                Ok(essay) => {
                    if cli.verbose {
                        divider("Essay");
                        info!("");
                        info!("  ══════════════════════════════════════");
                        info!("  📖  {}", essay.title);
                        info!("  ══════════════════════════════════════");
                    } else {
                        println!("{}\n", essay.title);
                    }
                    let count = essay.paragraphs.len();
                    for (i, para) in essay.paragraphs.iter().enumerate() {
                        let enhanced = maybe_enhance(para, stochastic, probability);
                        if cli.verbose {
                            let label = if i == 0 {
                                "Introduction".to_string()
                            } else if i == count - 1 {
                                "Conclusion".to_string()
                            } else {
                                format!("Body · §{}", i)
                            };
                            divider(&label);
                            info!("");
                            info!("  {}", enhanced);
                            info!("");
                        } else {
                            println!("{}\n", enhanced);
                        }
                    }
                }
                Err(e) => {
                    error!("  ❌ Essay generation failed: {}", e);
                }
            }
        }

        #[cfg(feature = "net")]
        Ask {
            prompt,
            limit,
            sentences,
            region,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Ask · Internet-Aware Knowledge Synthesis", "🌐");
            info!("  ❓ Prompt : {:?}", prompt);
            let aggregator = SearchAggregator::new().with_region(&region);
            divider("DuckDuckGo Results");

            if let Err(e) = aggregator.search_and_display(&prompt, limit).await {
                error!("  ⚠️  Search display failed: {}", e);
            }

            let corpus = match aggregator.get_response(&prompt).await {
                Ok(resp) => corpus_from_response(&resp),
                Err(_) => String::new(),
            };

            let final_corpus = if corpus.trim().is_empty() {
                let lite_results = aggregator.fetch(&prompt, limit).await.unwrap_or_default();
                let quality = corpus_from_results(&lite_results);
                if quality.trim().is_empty() {
                    corpus_from_results_raw(&lite_results)
                } else {
                    quality
                }
            } else {
                corpus
            };

            info!("");
            divider("LMM Response");

            if final_corpus.trim().is_empty() {
                error!("  ❌ No extractable content from search results.");
            } else {
                let summarizer = TextSummarizer::new(sentences, iterations, depth);
                match summarizer.summarize_with_query(&final_corpus, &prompt) {
                    Ok(summary) => {
                        for sentence in &summary {
                            let enhanced = maybe_enhance(sentence, stochastic, probability);
                            if cli.verbose {
                                info!("  {}", enhanced);
                            } else {
                                println!("{}", enhanced);
                            }
                        }
                    }
                    Err(e) => error!("  ❌ Summarization failed: {}", e),
                }
            }
        }

        Imagen {
            prompt,
            width,
            height,
            components,
            style,
            palette,
            output,
        } => {
            banner("Imagen · Spectral Field Synthesis", "🎨");
            info!("  🖼  Prompt    : {:?}", prompt);
            info!("  📐 Dimensions : {}x{}", width, height);
            info!("  🎭 Style      : {}", style);
            info!("  🎨 Palette    : {}", palette);
            info!("  🌊 Components : {}", components);
            let parsed_style = style.parse::<StyleMode>().unwrap_or(StyleMode::Plasma);
            let params = ImagenParams {
                prompt,
                width,
                height,
                components,
                style: parsed_style,
                palette_name: palette,
                output: output.clone(),
            };
            info!("");
            divider("Rendering");
            match imagen_render(&params) {
                Ok(path) => {
                    if cli.verbose {
                        info!("  ✅ Saved to: {}", path);
                    } else {
                        println!("{}", path);
                    }
                }
                Err(e) => error!("  ❌ Render failed: {}", e),
            }
        }
    }

    Ok(())
}
