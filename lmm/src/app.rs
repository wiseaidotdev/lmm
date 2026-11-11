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
            code_responder::CodeResponder,
            knowledge::{KnowledgeIndex, KnowledgeSource, ingest as cognition_ingest},
            r#loop::ThinkLoop,
            response_framer::ResponseFramer,
            search::SearchOracle,
            task_decomposer::{TaskDecomposer, TaskKind},
            thought::AdaptiveThoughtEngine,
            thought::classifier::InputClassifier,
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
    max_tasks: usize,
    no_frame: bool,
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

    let prompt_mode = InputClassifier::classify(&prompt);
    let sub_tasks = TaskDecomposer::decompose(&prompt, prompt_mode, max_tasks);
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

    let topic = extract_topic(&prompt);
    let kb_query = derive_kb_query(&prompt, prompt_mode, &topic);
    let top_k = 6 + (run_seed % 5) as usize;
    let kb_chunks: Vec<String> = index
        .query(&kb_query, top_k)
        .into_iter()
        .map(|c| c.text.clone())
        .collect();

    let (answer_text, from_kb) = match index.answer(&kb_query, top_k) {
        Some(answer) if is_quality_answer(&answer) => {
            let intent_ok = is_answer_intent_relevant(&answer, prompt_mode, &prompt);
            if intent_ok {
                info!("  ✅ KB answer found ({} chars)", answer.len());
                let prose = assemble_paragraphs_for(&answer, run_seed, &kb_query);
                (prose, true)
            } else {
                info!("  🔀 KB answer not intent-relevant. Building symbolic response");
                let text = build_symbolic_response(&prompt, &cold_corpus);
                (text, false)
            }
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
    let effective_prob = if !from_kb {
        0.0
    } else {
        match probability {
            Some(p) => p.clamp(0.0, 1.0),
            None => 0.05,
        }
    };
    let enhanced = StochasticEnhancer::new(effective_prob)
        .curated_only()
        .enhance_detailed(&answer_text);
    info!(
        "  🎲 substitutions={} language={:?} prob={:.2}",
        enhanced.substitutions, enhanced.language, effective_prob
    );

    let final_text = if no_frame {
        enhanced.text.clone()
    } else if TaskDecomposer::is_creative_task(&prompt) {
        let lang =
            TaskDecomposer::extract_code_language(&prompt).unwrap_or_else(|| "text".to_string());
        let kb_corpus: String = kb_chunks.join(" ");
        let task_desc = topic.clone();
        let code = CodeResponder::generate(&lang, &task_desc, &kb_corpus);
        let code_block = CodeResponder::format_code_block(&lang, &code);
        let frame = ResponseFramer::opening_phrase(prompt_mode, &topic, &prompt, &enhanced.text);
        if frame.is_empty() {
            format!("{}\n\n{}", enhanced.text.trim(), code_block)
        } else {
            format!("{}\n\n{}", frame.trim_end(), code_block)
        }
    } else {
        use crate::cognition::thought::classifier::ReasoningMode;
        let framing_topic = derive_framing_topic(&prompt, prompt_mode, &topic);
        let framed = ResponseFramer::frame(prompt_mode, &framing_topic, &prompt, &enhanced.text);
        if matches!(prompt_mode, ReasoningMode::Analogical) {
            restructure_analogical_response(&framed, &topic, run_seed)
        } else {
            framed
        }
    };

    if verbose {
        divider("TRAM Response");
        info!("  {}", final_text);
    } else {
        println!("{}", final_text);
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

/// Checks whether a KB answer is sufficiently relevant to the user's intent.
///
/// For evaluative and causal prompts the Wikipedia-style prose often provides
/// only general/historical facts.  When the answer does not contain at least
/// one intent-signal word we fall back to the symbolic response so the framing
/// opener does not dangle over unrelated content.
///
/// For other modes the function always returns `true` (pass-through).
///
/// # Time Complexity
///
/// O(|answer|)
#[cfg(feature = "cli")]
fn is_answer_intent_relevant(
    answer: &str,
    mode: crate::cognition::thought::classifier::ReasoningMode,
    prompt: &str,
) -> bool {
    use crate::cognition::thought::classifier::ReasoningMode;
    let lower = answer.to_ascii_lowercase();
    let prompt_lower = prompt.to_ascii_lowercase();
    match mode {
        ReasoningMode::Evaluative => {
            let wants_cons = prompt_lower.contains("inconvenience")
                || prompt_lower.contains("disadvantage")
                || prompt_lower.contains("limitation")
                || prompt_lower.contains("drawback")
                || prompt_lower.contains("weakness")
                || prompt_lower.contains("problem");
            if wants_cons {
                lower.contains("disadvantage")
                    || lower.contains("limitation")
                    || lower.contains("drawback")
                    || lower.contains("problem")
                    || lower.contains("slow")
                    || lower.contains("difficult")
                    || lower.contains("complex")
                    || lower.contains("challenge")
                    || lower.contains("overhead")
            } else {
                lower.contains("advantage")
                    || lower.contains("benefit")
                    || lower.contains("feature")
                    || lower.contains("performance")
                    || lower.contains("fast")
                    || lower.contains("safe")
                    || lower.contains("memory safe")
                    || lower.contains("efficient")
                    || lower.contains("speed")
                    || lower.contains("reliable")
                    || lower.contains("robust")
                    || lower.contains("concurrent")
                    || lower.contains("zero-cost")
                    || lower.contains("type safe")
                    || lower.contains("emphasizes performance")
                    || lower.contains("emphasizes")
            }
        }
        ReasoningMode::Causal => {
            lower.contains(" because ")
                || lower.contains(" since ")
                || lower.contains(" due to ")
                || lower.contains(" in order to ")
                || lower.contains(" to avoid ")
                || lower.contains(" to prevent ")
                || lower.contains(" to ensure ")
                || lower.contains("ownership")
                || lower.contains("borrow")
                || lower.contains("memory")
                || lower.contains("lifetime")
                || lower.contains("garbage")
                || lower.contains("design")
                || lower.contains("runtime")
                || lower.contains("overhead")
                || lower.contains("deterministic")
                || lower.contains("compile")
                || lower.contains("zero-cost")
        }
        _ => true,
    }
}

/// Assembles and shapes final output paragraphs, optionally filtering sentences
/// that open with an off-topic proper noun not related to `topic`.
///
/// For analogical topics such as `"rust vs python"`, sentences about either
/// entity are allowed through the proper-noun filter so both sides of the
/// comparison appear in the output.
#[cfg(feature = "cli")]
fn assemble_paragraphs_for(text: &str, run_seed: u64, topic: &str) -> String {
    let topic_cleaned = extract_topic(topic);
    let topic_ref = if topic_cleaned.is_empty() {
        topic
    } else {
        topic_cleaned.as_str()
    };

    static VS_SPLITTERS: &[&str] = &[" vs ", " versus "];
    let topic_ref_lower = topic_ref.to_ascii_lowercase();
    let topic_words: Vec<String> = {
        let mut words: Vec<String> = Vec::new();
        let mut is_comparison = false;
        for &splitter in VS_SPLITTERS {
            if let Some(pos) = topic_ref_lower.find(splitter) {
                let a = topic_ref[..pos].trim().to_ascii_lowercase();
                let b = topic_ref[pos + splitter.len()..]
                    .trim()
                    .to_ascii_lowercase();
                let a_first = a.split_whitespace().next().unwrap_or("").to_string();
                let b_first = b.split_whitespace().next().unwrap_or("").to_string();
                if !a_first.is_empty() {
                    words.push(a_first);
                }
                if !b_first.is_empty() {
                    words.push(b_first);
                }
                is_comparison = true;
                break;
            }
        }
        if !is_comparison {
            let first = topic_ref
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
            if !first.is_empty() {
                words.push(first);
            }
        }
        words
    };

    let prompt_lower = topic.to_ascii_lowercase();
    let query_keywords: Vec<&str> = prompt_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    let sentences: Vec<String> = split_sentences_with_punct(text)
        .into_iter()
        .map(strip_discourse_connector)
        .filter(|s| SentenceFilter::is_acceptable(s))
        .filter(|s| {
            static SITE_NAMES: &[&str] = &[
                "geeksforgeeks",
                "stackoverflow",
                "reddit",
                "medium",
                "github",
                "wikipedia",
                "quora",
                "hackernews",
                "dev.to",
                "towardsdatascience",
                "freecodecamp",
                "programiz",
                "tutorialspoint",
            ];
            let trimmed = s.trim();
            if trimmed.starts_with('-') || trimmed.starts_with('•') || trimmed.starts_with('*') {
                return false;
            }
            let lower = trimmed.to_ascii_lowercase();
            for site in SITE_NAMES {
                if lower.contains(site) {
                    return false;
                }
            }
            true
        })
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
            if topic_words.is_empty() {
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
            let s_lower = s.to_lowercase();
            for tw in &topic_words {
                if first_lower.contains(tw.as_str()) || tw.contains(first_lower.as_str()) {
                    return true;
                }
                if s_lower.contains(tw.as_str()) {
                    return true;
                }
            }
            false
        })
        .collect();

    if sentences.is_empty() {
        return text.to_string();
    }

    /// Scores a sentence by keyword overlap with the topic/query keywords.
    /// Sentences containing more of the query keywords score higher.
    /// History/metadata sentences (founding dates, sponsorship) score lower.
    fn score_sentence(s: &str, keywords: &[&str]) -> i32 {
        let lower = s.to_ascii_lowercase();
        let mut score = 0i32;
        for kw in keywords {
            if lower.contains(kw) {
                score += 2;
            }
        }
        static HISTORY_MARKERS: &[&str] = &[
            " created ",
            " founded ",
            " started ",
            " began ",
            " sponsored ",
            " announced ",
            " published ",
            "stable release",
            " first release",
            " major revision",
            " successor to ",
            "in the late ",
            "in 2006",
            "in 2007",
            "in 2008",
            "in 2009",
            "in 2010",
            "in 2011",
            "in 2012",
            "in 2013",
            "in 2014",
            "in 2015",
            "in 2016",
            "in 2017",
            "in 2018",
            "in 2019",
            "in 2020",
            "in 2021",
            "in 2022",
            "in 2023",
            "in 2024",
            "in 2025",
        ];
        for marker in HISTORY_MARKERS {
            if lower.contains(marker) {
                score -= 3;
            }
        }
        score
    }

    let mut scored: Vec<(i32, String)> = sentences
        .into_iter()
        .map(|s| (score_sentence(&s, &query_keywords), s))
        .collect();

    scored.sort_by_key(|a| std::cmp::Reverse(a.0));

    let best_score = scored.first().map(|(sc, _)| *sc).unwrap_or(0);
    let top_tier: Vec<String> = scored
        .iter()
        .filter(|(sc, _)| *sc >= best_score)
        .map(|(_, s)| s.clone())
        .collect();
    let rest: Vec<String> = scored
        .iter()
        .filter(|(sc, _)| *sc < best_score)
        .map(|(_, s)| s.clone())
        .collect();

    let mut top_shuffled = top_tier;
    let top_n = top_shuffled.len();
    let mut seed = run_seed;
    for i in (1..top_n).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        top_shuffled.swap(i, j);
    }

    let mut rest_shuffled = rest;
    let rest_n = rest_shuffled.len();
    for i in (1..rest_n).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        rest_shuffled.swap(i, j);
    }

    top_shuffled.extend(rest_shuffled);
    let sentences = top_shuffled;

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

/// Derives the most relevant KB query string for a given reasoning mode and prompt.
///
/// For `Evaluative` mode, focuses on advantages, features, or drawbacks depending
/// on the prompt's intent. For `Causal` mode, strips the question prefix so the
/// predicate phrase is used for TF-IDF retrieval. Otherwise returns the original
/// prompt unchanged.
///
/// # Examples
///
/// ```
/// use lmm::cognition::thought::classifier::ReasoningMode;
/// let q = derive_kb_query("what are the advantages of rust?", ReasoningMode::Evaluative, "rust");
/// assert!(q.contains("advantages") || q.contains("benefits"));
/// ```
///
/// # Time Complexity
///
/// O(|prompt|)
#[cfg(feature = "cli")]
fn derive_kb_query(
    prompt: &str,
    mode: crate::cognition::thought::classifier::ReasoningMode,
    topic: &str,
) -> String {
    use crate::cognition::thought::classifier::ReasoningMode;
    match mode {
        ReasoningMode::Evaluative => {
            let pl = prompt.to_ascii_lowercase();
            if pl.contains("inconvenience")
                || pl.contains("disadvantage")
                || pl.contains("limitation")
                || pl.contains("drawback")
                || pl.contains("weakness")
                || pl.contains("problem")
                || pl.contains("downside")
            {
                format!("{} disadvantages limitations problems drawbacks", topic)
            } else {
                format!("{} advantages benefits features performance", topic)
            }
        }
        ReasoningMode::Causal => {
            static CAUSAL_OPENERS: &[&str] = &[
                "why does ",
                "why do ",
                "why is ",
                "why are ",
                "why was ",
                "why would ",
                "why should ",
                "why can't ",
                "why cannot ",
                "why doesn't ",
                "why don't ",
                "how come ",
            ];
            let lower = prompt.to_ascii_lowercase();
            let lower_clean = lower.trim_end_matches('?').trim_end_matches('.').trim();
            let predicate = CAUSAL_OPENERS
                .iter()
                .find_map(|&pfx| lower_clean.strip_prefix(pfx))
                .unwrap_or(lower_clean)
                .trim();
            if predicate.len() > topic.len() + 2 {
                predicate.to_string()
            } else {
                prompt.to_string()
            }
        }
        _ => prompt.to_string(),
    }
}

/// Restructures an analogical (comparison) response so each entity's sentences
/// appear in a dedicated paragraph separated by a natural-language connector
/// phrase (e.g. `"On the other hand,"`).
///
/// The function extracts the opening sentence (the framed opener such as
/// `"Comparing Rust and Python: the key differences are as follows."`) and
/// then groups the remaining sentences by which comparison entity they discuss.
/// Entity-A sentences form the first paragraph; Entity-B sentences form the
/// second, preceded by a connector.  Sentences that mention both entities
/// (or neither) are placed in the first paragraph.
///
/// When only one entity is mentioned throughout, the original `framed` text is
/// returned unchanged to avoid an empty second paragraph.
///
/// # Time Complexity
///
/// O(|framed|)
#[cfg(feature = "cli")]
fn restructure_analogical_response(framed: &str, topic: &str, run_seed: u64) -> String {
    static VS_SPLITTERS: &[&str] = &[" vs ", " versus ", " compared to "];
    let topic_lower = topic.to_ascii_lowercase();
    let mut entity_a = String::new();
    let mut entity_b = String::new();
    for splitter in VS_SPLITTERS {
        if let Some(pos) = topic_lower.find(splitter) {
            entity_a = topic_lower[..pos].trim().to_string();
            entity_b = topic_lower[pos + splitter.len()..].trim().to_string();
            break;
        }
    }
    if entity_a.is_empty() || entity_b.is_empty() {
        return framed.to_string();
    }
    let opener_end = framed.find(". ").map(|i| i + 2).unwrap_or(framed.len());
    let opener = &framed[..opener_end.min(framed.len())];
    let body = framed[opener_end.min(framed.len())..].trim();
    if body.is_empty() {
        return framed.to_string();
    }
    let raw_sentences: Vec<String> = split_sentences_with_punct(body)
        .into_iter()
        .map(strip_discourse_connector)
        .filter(|s| SentenceFilter::is_acceptable(s))
        .collect();
    if raw_sentences.is_empty() {
        return framed.to_string();
    }
    let mut a_sentences: Vec<String> = Vec::new();
    let mut b_sentences: Vec<String> = Vec::new();
    let mut neutral: Vec<String> = Vec::new();
    for s in raw_sentences {
        let sl = s.to_ascii_lowercase();
        let mentions_a = sl.contains(entity_a.as_str());
        let mentions_b = sl.contains(entity_b.as_str());
        match (mentions_a, mentions_b) {
            (true, false) => a_sentences.push(s),
            (false, true) => b_sentences.push(s),
            _ => neutral.push(s),
        }
    }
    if b_sentences.is_empty() {
        return framed.to_string();
    }
    static CONNECTORS: &[&str] = &[
        "On the other hand,",
        "By contrast,",
        "Meanwhile,",
        "In comparison,",
        "Conversely,",
    ];
    let connector = CONNECTORS[run_seed as usize % CONNECTORS.len()];
    let mut para_a: Vec<String> = neutral;
    para_a.extend(a_sentences);
    let mut parts: Vec<String> = vec![opener.trim().to_string()];
    if !para_a.is_empty() {
        parts.push(para_a.join(" "));
    }
    let b_first = b_sentences[0].clone();
    let b_rest = b_sentences[1..].join(" ");
    let b_para = if b_rest.is_empty() {
        format!("{} {}", connector, b_first)
    } else {
        format!("{} {} {}", connector, b_first, b_rest)
    };
    parts.push(b_para);
    parts.join("\n\n")
}

/// Derives the best topic string to pass to [`ResponseFramer::frame`] for a
/// given reasoning mode.
///
/// For `Causal` mode a richer predicate fragment produces a more natural opening.
/// For example, `"why does rust not have a garbage collector?"` frames as
/// `"Rust does not have a garbage collector"` rather than just `"Rust"`.
///
/// # Time Complexity
///
/// O(|prompt|)
#[cfg(feature = "cli")]
fn derive_framing_topic(
    prompt: &str,
    mode: crate::cognition::thought::classifier::ReasoningMode,
    topic: &str,
) -> String {
    use crate::cognition::thought::classifier::ReasoningMode;
    if !matches!(mode, ReasoningMode::Causal) {
        return topic.to_string();
    }
    static CAUSAL_PREFIXES: &[(&str, &str)] = &[
        ("why does ", "does"),
        ("why do ", "do"),
        ("why is ", "is"),
        ("why are ", "are"),
        ("why was ", "was"),
        ("why were ", "were"),
        ("why would ", "would"),
        ("why should ", "should"),
        ("why can't ", "cannot"),
        ("why cannot ", "cannot"),
        ("why doesn't ", "does not"),
        ("why don't ", "do not"),
        ("how come ", ""),
    ];
    let lower = prompt.to_lowercase();
    for &(prefix, aux) in CAUSAL_PREFIXES {
        if lower.starts_with(prefix) {
            let rest = prompt[prefix.len()..]
                .trim()
                .trim_end_matches('?')
                .trim_end_matches('.')
                .trim();
            if rest.len() > 2 {
                let words: Vec<&str> = rest.split_whitespace().collect();
                let first_word_lower = words
                    .first()
                    .map(|w| w.to_ascii_lowercase())
                    .unwrap_or_default();
                let needs_aux_insertion = !aux.is_empty()
                    && ![
                        "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
                        "will", "would", "could", "should", "can", "cannot", "doesn't", "don't",
                        "can't", "won't", "wouldn't", "couldn't", "not",
                    ]
                    .contains(&first_word_lower.as_str())
                    && words.len() >= 2;
                let reconstructed = if needs_aux_insertion {
                    let mut subject_words: Vec<&str> = Vec::new();
                    let mut rest_words: Vec<&str> = Vec::new();
                    let mut past_subject = false;
                    static PREDICATE_STARTERS: &[&str] = &[
                        "not", "no", "never", "have", "has", "had", "make", "makes", "work",
                        "works", "allow", "allows", "prevent", "prevents", "require", "requires",
                        "cause", "causes", "use", "uses", "need", "needs",
                    ];
                    for &w in &words {
                        if !past_subject
                            && PREDICATE_STARTERS.contains(&w.to_ascii_lowercase().as_str())
                        {
                            past_subject = true;
                        }
                        if past_subject {
                            rest_words.push(w);
                        } else {
                            subject_words.push(w);
                        }
                    }
                    if subject_words.is_empty() {
                        rest.to_string()
                    } else {
                        let subject = subject_words.join(" ");
                        let predicate = rest_words.join(" ");
                        if predicate.is_empty() {
                            format!("{} {}", subject, aux)
                        } else {
                            format!("{} {} {}", subject, aux, predicate)
                        }
                    }
                } else {
                    rest.to_string()
                };
                let mut chars = reconstructed.chars();
                let first_up: String = chars
                    .next()
                    .map(|c| c.to_uppercase().collect::<String>())
                    .unwrap_or_default();
                return format!("{}{}", first_up, chars.as_str());
            }
        }
    }
    topic.to_string()
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
        "write a ",
        "write an ",
        "write ",
        "generate a ",
        "generate an ",
        "create a ",
        "create an ",
        "implement a ",
        "implement an ",
        "code a ",
        "build a ",
        "show me how to ",
        "show me a ",
        "show me ",
        "give me a ",
        "give me an ",
    ];
    static MODE_KEYWORD_PREFIXES: &[&str] = &[
        "advantages of ",
        "advantage of ",
        "benefits of ",
        "benefit of ",
        "disadvantages of ",
        "disadvantage of ",
        "pros and cons of ",
        "pros of ",
        "cons of ",
        "drawbacks of ",
        "drawback of ",
        "strengths of ",
        "strength of ",
        "weaknesses of ",
        "weakness of ",
        "limitations of ",
        "limitation of ",
        "inconveniences of ",
        "inconvenience of ",
        "upsides of ",
        "downsides of ",
        "merits of ",
        "demerits of ",
        "causes of ",
        "cause of ",
        "reason for ",
        "reasons for ",
        "explanation of ",
        "effect of ",
        "effects of ",
        "definition of ",
        "meaning of ",
        "concept of ",
        "examples of ",
        "overview of ",
        "history of ",
        "installation of ",
        "setup of ",
        "usage of ",
        "use of ",
        "uses of ",
        "comparison of ",
        "difference between ",
        "differences between ",
        "similarities between ",
        "relationship between ",
    ];
    let lower = prompt.to_lowercase();
    let after_filler: Option<&str> = FILLER_PREFIXES
        .iter()
        .find(|&&prefix| lower.starts_with(prefix))
        .map(|&prefix| &prompt[prefix.len()..]);
    let raw_after_filler = if let Some(r) = after_filler {
        r.trim()
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim_end_matches('!')
            .trim()
    } else {
        prompt
            .trim()
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim_end_matches('!')
            .trim()
    };
    let raw_lower = raw_after_filler.to_ascii_lowercase();
    let after_mode = MODE_KEYWORD_PREFIXES
        .iter()
        .find(|&&pfx| raw_lower.starts_with(pfx))
        .map(|&pfx| raw_after_filler[pfx.len()..].trim())
        .unwrap_or(raw_after_filler);
    let stripped = strip_leading_articles(after_mode);
    if after_filler.is_some() {
        strip_trailing_verb(trim_predicate(&stripped))
    } else {
        stripped
    }
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

/// Fires a lightweight intent-specific DDG search and ingests snippets into `index`.
///
/// Unlike `prefetch_web` this never skips on `kb_provided`; it is meant for
/// secondary enrichment queries generated by [`crate::cognition::task_decomposer::TaskDecomposer`].
///
/// # Arguments
///
/// * `query`   - The search query string.
/// * `index`   - The `KnowledgeIndex` to populate.
/// * `verbose` - When `true`, diagnostic lines are printed.
#[cfg(feature = "cli")]
async fn prefetch_web_for_query(query: &str, index: &mut KnowledgeIndex, verbose: bool) {
    prefetch_web_for_query_inner(query, index, verbose).await;
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
    let topic_advantages = format!("advantages of {}", topic_query);
    let topic_pros_cons = format!("pros and cons of {}", topic_query);
    let topic_how_to = format!("how to {}", topic_query);
    let topic_why = format!("why {}", topic_query);
    let topic_vs = topic_query.clone();

    let topic_words: Vec<String> = topic_query
        .split_whitespace()
        .filter(|w| w.len() >= 4)
        .map(|w| w.to_string())
        .collect();

    let prompt_lower = prompt.to_ascii_lowercase();
    let is_evaluative_prompt = prompt_lower.contains("advantage")
        || prompt_lower.contains("benefit")
        || prompt_lower.contains("pros")
        || prompt_lower.contains("disadvantage")
        || prompt_lower.contains("inconvenience")
        || prompt_lower.contains("limitation")
        || prompt_lower.contains("drawback");
    let is_causal_prompt = prompt_lower.starts_with("why ")
        || prompt_lower.starts_with("how come")
        || prompt_lower.contains(" cause")
        || prompt_lower.contains(" reason");
    let is_procedural_prompt = prompt_lower.starts_with("how to ")
        || prompt_lower.contains("install ")
        || prompt_lower.contains(" setup")
        || prompt_lower.contains("tutorial");
    let is_analogical_prompt = topic_query.to_ascii_lowercase().contains(" vs ")
        || topic_query.to_ascii_lowercase().contains(" versus ")
        || prompt_lower.contains(" vs ")
        || prompt_lower.contains(" versus ");

    let causal_predicate_query: Option<String> = if is_causal_prompt {
        static CAUSAL_OPENERS: &[&str] = &[
            "why does ",
            "why do ",
            "why is ",
            "why are ",
            "why was ",
            "why were ",
            "why would ",
            "why should ",
            "why can't ",
            "why cannot ",
            "why doesn't ",
            "why don't ",
            "how come ",
        ];
        let lower_clean = prompt_lower
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim();
        let predicate = CAUSAL_OPENERS
            .iter()
            .find_map(|&pfx| lower_clean.strip_prefix(pfx))
            .unwrap_or(lower_clean)
            .trim();
        if predicate.len() > topic_query.len() + 2 {
            let words: Vec<&str> = predicate.split_whitespace().collect();
            let query_words = if words.len() > 7 { &words[..7] } else { &words };
            Some(query_words.join(" "))
        } else {
            None
        }
    } else {
        None
    };

    let (entity_lang_a, entity_lang_b) = if is_analogical_prompt {
        static VS_SPLITTERS: &[&str] = &[" vs ", " versus "];
        let tq_lower = topic_query.to_ascii_lowercase();
        let mut pair: (Option<String>, Option<String>) = (None, None);
        for &splitter in VS_SPLITTERS {
            if let Some(pos) = tq_lower.find(splitter) {
                let a = topic_query[..pos].trim().to_string();
                let b = topic_query[pos + splitter.len()..].trim().to_string();
                if !a.is_empty() && !b.is_empty() {
                    pair = (
                        Some(format!("{} programming language", a)),
                        Some(format!("{} programming language", b)),
                    );
                    break;
                }
            }
        }
        pair
    } else {
        (None, None)
    };

    let mut ia_candidates: Vec<&str> = Vec::new();
    if let Some(ref ea) = entity_lang_a {
        ia_candidates.push(ea.as_str());
    }
    if let Some(ref eb) = entity_lang_b {
        ia_candidates.push(eb.as_str());
    }
    ia_candidates.push(topic_query.as_str());
    ia_candidates.push(topic_lang.as_str());
    if is_evaluative_prompt {
        ia_candidates.push(topic_advantages.as_str());
        ia_candidates.push(topic_pros_cons.as_str());
    }
    if is_causal_prompt {
        ia_candidates.push(topic_why.as_str());
        if let Some(ref cpq) = causal_predicate_query
            && !ia_candidates.contains(&cpq.as_str())
        {
            ia_candidates.push(cpq.as_str());
        }
    }
    if is_procedural_prompt {
        ia_candidates.push(topic_how_to.as_str());
    }
    ia_candidates.extend_from_slice(&[
        topic_lang2.as_str(),
        topic_library.as_str(),
        topic_framework.as_str(),
        topic_what.as_str(),
        topic_vs.as_str(),
        prompt,
    ]);
    for tw in &topic_words {
        if !ia_candidates.contains(&tw.as_str()) {
            ia_candidates.push(tw.as_str());
        }
    }

    let analogical_entity_queries: Vec<&str> = {
        let mut v = Vec::new();
        if let Some(ref ea) = entity_lang_a {
            v.push(ea.as_str());
        }
        if let Some(ref eb) = entity_lang_b {
            v.push(eb.as_str());
        }
        v
    };

    let must_fetch_queries: Vec<&str> = {
        let mut v = analogical_entity_queries.clone();
        if let Some(ref cpq) = causal_predicate_query
            && !v.contains(&cpq.as_str())
        {
            v.push(cpq.as_str());
        }
        v
    };

    let mut ia_winner: Option<&str> = None;

    for query in ia_candidates.iter().copied() {
        let is_must_fetch = must_fetch_queries.contains(&query);
        if prose_chunks > 0 && !is_must_fetch {
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
        let needs_specific_content = is_evaluative_prompt || is_causal_prompt;
        if !needs_specific_content {
            return;
        }
        info!("  🌐 Supplementing with intent-specific Lite search");
    }

    if ingested > 0 && prose_chunks == 0 {
        info!(
            "  🌐 Instant Answer [{:?}]: {} fragment(s), augmenting with Lite search",
            ia_winner.unwrap_or(topic_query.as_str()),
            ingested
        );
    }

    let mut lite_snippets_found = 0usize;
    let mut lite_q_list: Vec<String> = Vec::new();
    let full_prompt_clean = prompt
        .trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim()
        .to_string();
    if let Some(ref cpq) = causal_predicate_query {
        lite_q_list.push(cpq.clone());
        let topic_no_gc = format!("{} memory management", topic_query);
        lite_q_list.push(topic_no_gc);
        let topic_ownership = format!("{} ownership model", topic_query);
        lite_q_list.push(topic_ownership);
    }
    if is_evaluative_prompt {
        lite_q_list.push(format!("advantages of {}", topic_query));
        lite_q_list.push(format!("{} pros cons", topic_query));
        lite_q_list.push(format!("benefits of {}", topic_query));
    }
    for eq in [entity_lang_a.as_deref(), entity_lang_b.as_deref()]
        .iter()
        .flatten()
    {
        if !lite_q_list.iter().any(|q| q.as_str() == *eq) {
            lite_q_list.push((*eq).to_string());
        }
    }

    lite_q_list.push(full_prompt_clean.clone());
    if topic_query.as_str() != prompt {
        lite_q_list.push(topic_query.clone());
    }
    if !lite_q_list.iter().any(|q| q.as_str() == prompt) {
        lite_q_list.push(prompt.to_string());
    }
    let topic_words_significant: Vec<&str> = topic_query
        .split_whitespace()
        .filter(|w| {
            w.len() >= 4
                && !matches!(
                    w.to_ascii_lowercase().as_str(),
                    "what" | "does" | "have" | "that" | "this" | "with" | "from" | "into"
                )
        })
        .collect();
    if topic_words_significant.len() >= 2 {
        for i in 0..topic_words_significant.len().saturating_sub(1) {
            let pair = format!(
                "{} {}",
                topic_words_significant[i],
                topic_words_significant[i + 1]
            );
            if !lite_q_list.iter().any(|q| q == &pair) {
                lite_q_list.push(pair);
            }
        }
    }
    for word in &topic_words_significant {
        if !lite_q_list.iter().any(|q| q.as_str() == *word) {
            lite_q_list.push((*word).to_string());
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
                    let was_supplementing = prose_chunks > 0;
                    if !was_supplementing {
                        break 'lite;
                    }
                }
            }
            Ok(_) => {
                if verbose {
                    info!("  🌐 Lite search [{:?}]: no results", lite_q.as_str());
                }
            }
            Err(_e) => {
                if verbose {
                    info!(
                        "  ⚠️  Lite search [{:?}]: network error, trying IA fallback",
                        lite_q.as_str()
                    );
                }
                let encoded: String = lite_q
                    .chars()
                    .map(|c| if c == ' ' { '+' } else { c })
                    .collect();
                let ia_path = format!("?q={}&ia=web", encoded);
                if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
                    let mut ia_added = 0usize;
                    if let Some(text) = resp
                        .abstract_text
                        .filter(|s| s.split_whitespace().count() >= 6)
                    {
                        ia_added += index.ingest_text("web-ia-fallback", text.trim());
                    }
                    if let Some(def) = resp
                        .definition
                        .filter(|s| s.split_whitespace().count() >= 6)
                    {
                        ia_added += index.ingest_text("web-ia-fallback", def.trim());
                    }
                    if ia_added > 0 {
                        if verbose {
                            info!(
                                "  🌐 IA fallback [{:?}]: {} chunk(s) ingested",
                                lite_q.as_str(),
                                ia_added
                            );
                        }
                        ingested += ia_added;
                        lite_snippets_found += ia_added;
                        let was_supplementing = prose_chunks > 0;
                        if !was_supplementing {
                            break 'lite;
                        }
                    } else {
                        break 'lite;
                    }
                } else {
                    break 'lite;
                }
                let encoded: String = lite_q
                    .chars()
                    .map(|c| if c == ' ' { '+' } else { c })
                    .collect();
                let ia_path = format!("?q={}&ia=web", encoded);
                if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
                    let mut ia_added = 0usize;
                    if let Some(text) = resp
                        .abstract_text
                        .filter(|s| s.split_whitespace().count() >= 6)
                    {
                        ia_added += index.ingest_text("web-ia-fallback", text.trim());
                    }
                    if let Some(def) = resp
                        .definition
                        .filter(|s| s.split_whitespace().count() >= 6)
                    {
                        ia_added += index.ingest_text("web-ia-fallback", def.trim());
                    }
                    if ia_added > 0 {
                        if verbose {
                            info!(
                                "  🌐 IA fallback [{:?}]: {} chunk(s) ingested",
                                lite_q.as_str(),
                                ia_added
                            );
                        }
                        ingested += ia_added;
                        lite_snippets_found += ia_added;
                        let was_supplementing = prose_chunks > 0;
                        if !was_supplementing {
                            break 'lite;
                        }
                    } else {
                        break 'lite;
                    }
                } else {
                    break 'lite;
                }
            }
        }
    }
    if lite_snippets_found == 0 && ingested == 0 {
        info!("  🌐 Lite search: no results");
        let ducky_q = format!("!ducky+{}", topic_query.replace(' ', "+"));
        if let Ok(ducky_results) = browser.lite_search(&ducky_q, "wt-wt", Some(5), ua).await {
            let mut ducky_added = 0usize;
            for r in &ducky_results {
                let snippet = r.snippet.trim();
                let title = r.title.trim();
                if snippet.split_whitespace().count() >= 5 {
                    let chunk = format!("{} {}", title, snippet);
                    ducky_added += index.ingest_text("web-ducky", chunk.trim());
                }
            }
            if ducky_added > 0 {
                info!("  🌐 !ducky fallback: {} chunk(s) ingested", ducky_added);
            }
        }
    }
}

#[cfg(all(feature = "cli", feature = "net"))]
async fn prefetch_web_for_query_inner(query: &str, index: &mut KnowledgeIndex, verbose: bool) {
    let browser = Browser::new();
    let ua = get_ua("firefox").unwrap_or("Mozilla/5.0");
    let encoded: String = query
        .chars()
        .map(|c| if c == ' ' { '+' } else { c })
        .collect();
    let ia_path = format!("?q={}&ia=web", encoded);
    let mut ingested = 0usize;
    if let Ok(resp) = browser.get_api_response(&ia_path, None).await {
        if let Some(text) = resp.abstract_text.filter(|s| !s.trim().is_empty()) {
            ingested += index.ingest_text("web-intent", text.trim());
        }
        if let Some(def) = resp.definition.filter(|s| !s.trim().is_empty()) {
            ingested += index.ingest_text("web-intent", def.trim());
        }
    }
    if ingested == 0
        && let Ok(results) = browser.lite_search(query, "wt-wt", Some(10), ua).await
    {
        for r in &results {
            let snippet = r.snippet.trim();
            let title = r.title.trim();
            if snippet.split_whitespace().count() >= 8 {
                let chunk = format!("{} {}", title, snippet);
                ingested += index.ingest_text("web-intent", chunk.trim());
            }
        }
    }
    if ingested == 0 {
        let ducky_q = format!("!ducky+{}", query.replace(' ', "+"));
        if let Ok(ducky_results) = browser.lite_search(&ducky_q, "wt-wt", Some(5), ua).await {
            for r in &ducky_results {
                let snippet = r.snippet.trim();
                let title = r.title.trim();
                if snippet.split_whitespace().count() >= 5 {
                    let chunk = format!("{} {}", title, snippet);
                    ingested += index.ingest_text("web-ducky", chunk.trim());
                }
            }
            if verbose && ingested > 0 {
                info!(
                    "  🌐 !ducky fallback [{:?}]: {} chunk(s) ingested",
                    query, ingested
                );
            }
        }
    } else if verbose {
        info!(
            "  🌐 Intent query [{:?}]: {} chunk(s) ingested",
            query, ingested
        );
    }
}

#[cfg(all(feature = "cli", not(feature = "net")))]
async fn prefetch_web_for_query_inner(_query: &str, _index: &mut KnowledgeIndex, _verbose: bool) {}

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
                cli.max_tasks,
                cli.no_frame,
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
            max_tasks,
            no_frame,
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
                max_tasks,
                no_frame,
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
