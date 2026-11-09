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
use crate::causal::CausalGraph;
#[cfg(all(feature = "cli", feature = "net"))]
use crate::cli::commands::Commands::Ask;
#[cfg(feature = "cli")]
use crate::cli::commands::{
    Cli,
    Commands::{
        Causal, Consciousness as CmdConsciousness, Decode, Discover, Encode, Essay,
        Field as CmdField, Imagen, Paragraph, Physics, Predict, Sentence, Simulate, Summarize,
        Tram,
    },
};
#[cfg(feature = "cli")]
use crate::cognition::knowledge::{KnowledgeIndex, KnowledgeSource, ingest as cognition_ingest};
#[cfg(feature = "cli")]
use crate::cognition::r#loop::ThinkLoop;
#[cfg(feature = "cli")]
use crate::cognition::search::SearchOracle;
#[cfg(feature = "cli")]
use crate::consciousness::Consciousness;
#[cfg(feature = "cli")]
use crate::discovery::SymbolicRegression;
#[cfg(feature = "cli")]
use crate::encode::{decode_message, encode_text};
#[cfg(feature = "cli")]
use crate::expert::inference::InferenceEngine;
#[cfg(feature = "cli")]
use crate::expert::rule::RuleBase;
#[cfg(feature = "cli")]
use crate::field::Field;
#[cfg(feature = "cli")]
use crate::generation::prose::ProseBuilder;
#[cfg(feature = "cli")]
use crate::lexicon::Lexicon;
#[cfg(feature = "cli")]
use crate::physics::{HarmonicOscillator, LorenzSystem, Pendulum, SIRModel};
#[cfg(feature = "cli")]
use crate::predict::TextPredictor;
#[cfg(feature = "cli")]
use crate::simulation::Simulator;
#[cfg(feature = "cli")]
use crate::stochastic::StochasticEnhancer;
#[cfg(feature = "cli")]
use crate::tensor::Tensor;
#[cfg(feature = "cli")]
use crate::text::{EssayGenerator, ParagraphGenerator, SentenceGenerator, TextSummarizer};
#[cfg(feature = "cli")]
use crate::traits::{Causal as CausalTrait, Simulatable};
#[cfg(feature = "cli")]
use clap::Parser;
#[cfg(all(feature = "cli", feature = "net"))]
use duckduckgo::browser::Browser;
#[cfg(all(feature = "cli", feature = "net"))]
use duckduckgo::user_agents::get as get_ua;
#[cfg(feature = "cli")]
use tracing::{Event, Subscriber, error, info};
#[cfg(feature = "cli")]
use tracing_appender::rolling;
#[cfg(feature = "cli")]
use tracing_subscriber::{
    Layer, Registry, filter,
    fmt::{self, FmtContext, FormatEvent, FormatFields},
    prelude::__tracing_subscriber_SubscriberExt,
    registry::LookupSpan,
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
        mut writer: fmt::format::Writer<'_>,
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

    let console_layer = fmt::Layer::new()
        .compact()
        .without_time()
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_target(false)
        .with_writer(std::io::stdout)
        .event_format(NoLevelFormatter)
        .with_filter(console_level);

    let file_layer = fmt::Layer::new()
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
fn load_source(input: &str, text: String) -> crate::error::Result<String> {
    if input == "-" {
        Ok(text)
    } else {
        std::fs::read_to_string(input)
            .map_err(|e| crate::error::LmmError::Perception(e.to_string()))
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
/// * `prompt`          - Natural-language question or task description.
/// * `knowledge_base`  - Optional filesystem path to a KB file or directory.
/// * `probability`     - Optional per-word synonym-substitution probability [0.0, 1.0].
/// * `verbose`         - Whether to emit rich diagnostic output.
/// * `iterations`      - Maximum ThinkLoop iterations (default 10).
/// * `inference_depth` - Backward-chaining depth limit for expert reasoning.
/// * `cf_threshold`    - Minimum certainty factor threshold for expert inference.
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
pub async fn handle_tram(
    prompt: String,
    knowledge_base: Option<String>,
    probability: Option<f64>,
    verbose: bool,
    iterations: usize,
    inference_depth: usize,
    cf_threshold: f64,
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
        let path = std::path::Path::new(&kb_path);
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

    divider("ThinkLoop Reasoning");

    let run_seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;

    let mut oracle = SearchOracle::new(5);
    let convergence_threshold = 0.70;
    let think_iterations = if index.is_empty() {
        iterations
    } else {
        iterations.min(3)
    };
    let mut lp = ThinkLoop::builder(&prompt)
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

    let top_k = 5 + (run_seed % 6) as usize;
    let kb_chunks: Vec<String> = index
        .query(&prompt, top_k)
        .into_iter()
        .map(|c| c.text.clone())
        .collect();

    let (answer_text, from_kb) = match index.answer(&prompt, top_k) {
        Some(answer) if is_quality_answer(&answer) => {
            info!("  ✅ KB answer found ({} chars)", answer.len());
            let prose = assemble_paragraphs(&answer, run_seed);
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
        if expert_prose.split_whitespace().count() >= 20 {
            expert_prose + "\n\n" + &answer_text
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
                0.12
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
    if word_count < 20 {
        return false;
    }
    let sentences: Vec<&str> = answer
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let sentence_count = sentences.len().max(1);
    let avg_words_per_sentence = word_count / sentence_count;

    if avg_words_per_sentence < 12 {
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
        })
        .count();
    verb_count * 10 >= sentence_count * 4
}

/// Assembles raw KB answer text into randomised multi-paragraph prose.
///
/// Sentences are split from the raw answer, shuffled pseudo-randomly with
/// `run_seed`, then grouped into 1-3 paragraphs separated by blank lines.
/// A discourse connector is prepended to the first sentence of each
/// subsequent paragraph so the output reads naturally.
///
/// # Arguments
///
/// * `text`     - Raw extractive answer from `KnowledgeIndex::answer`.
/// * `run_seed` - Per-run nanosecond entropy value from [`handle_tram`].
///
/// # Time Complexity
///
/// O(|text|)
#[cfg(feature = "cli")]
fn assemble_paragraphs(text: &str, run_seed: u64) -> String {
    static CONNECTORS: &[&str] = &[
        "Furthermore,",
        "In addition,",
        "Notably,",
        "Moreover,",
        "As a result,",
        "In practice,",
        "By extension,",
        "Interestingly,",
        "At its core,",
        "Consequently,",
    ];

    let mut sentences: Vec<String> = text
        .split(['.', '!', '?'])
        .map(|s| s.trim().to_string())
        .filter(|s| s.split_whitespace().count() >= 5)
        .map(|s| {
            let mut out = s.clone();
            if !out.ends_with('.') && !out.ends_with('!') && !out.ends_with('?') {
                out.push('.');
            }
            out
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

    let para_count = if sentences.len() <= 2 {
        1
    } else if sentences.len() <= 4 {
        1 + (run_seed % 2) as usize
    } else {
        1 + (run_seed % 3) as usize
    };

    let per_para = (sentences.len() / para_count).max(1);
    let mut paragraphs: Vec<String> = Vec::new();
    let mut offset = 0usize;

    for p in 0..para_count {
        let take = if p == para_count - 1 {
            sentences.len() - offset
        } else {
            per_para
        };
        if take == 0 {
            break;
        }
        let chunk: Vec<&str> = sentences[offset..offset + take]
            .iter()
            .map(|s| s.as_str())
            .collect();
        offset += take;

        let mut para = if p > 0 {
            let conn = CONNECTORS[(run_seed as usize + p) % CONNECTORS.len()];
            let first = chunk[0];
            let joined_first = if let Some(fc) = first.chars().next() {
                format!(
                    "{} {}{}",
                    conn,
                    fc.to_ascii_lowercase(),
                    &first[fc.len_utf8()..]
                )
            } else {
                format!("{} {}", conn, first)
            };
            let mut parts = vec![joined_first];
            parts.extend(chunk[1..].iter().map(|s| s.to_string()));
            parts.join(" ")
        } else {
            chunk.join(" ")
        };

        if let Some(fc) = para.chars().next() {
            let upper: String = fc.to_uppercase().collect();
            para = upper + &para[fc.len_utf8()..];
        }

        paragraphs.push(para);
    }

    paragraphs.join("\n\n")
}

/// Builds a topic-aware symbolic response when no KB answer is available.
///
/// First attempts to summarise the ThinkLoop cold-memory corpus. If that
/// corpus is too thin, falls back to `ParagraphGenerator` which uses
/// `keywords_from_text` to anchor sentences to the actual prompt topic.
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
    let para_gen = ParagraphGenerator::new(4, 20, 3);
    para_gen
        .generate(prompt)
        .unwrap_or_else(|_| prompt.to_string())
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
        "where is ",
        "when is ",
        "give me information about ",
        "give me info on ",
    ];
    let lower = prompt.to_lowercase();
    for prefix in FILLER_PREFIXES {
        if lower.starts_with(prefix) {
            let raw = prompt[prefix.len()..].trim();
            return strip_leading_articles(raw);
        }
    }
    strip_leading_articles(prompt.trim())
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

    let ia_candidates: &[&str] = &[
        topic_query.as_str(),
        topic_lang.as_str(),
        topic_lang2.as_str(),
        topic_what.as_str(),
        prompt,
    ];

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
                    if wc >= 20 {
                        let added = index.ingest_text("web-ia", text.trim());
                        prose_chunks += added;
                        ingested += added;
                    } else if wc >= 8 {
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
    match browser.lite_search(prompt, "wt-wt", Some(15), ua).await {
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
                    "  🌐 Lite search: {} results (titles only - no snippets)",
                    results.len()
                );
            } else {
                info!(
                    "  🌐 Lite search: {} result(s), {} snippet chunk(s) ingested",
                    results.len(),
                    ingested
                );
            }
        }
        Ok(_) => {
            if verbose {
                info!("  🌐 Lite search: no results");
            }
        }
        Err(e) => {
            if verbose {
                info!("  ⚠️  Lite search error: {e}");
            }
        }
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
                std::process::exit(1);
            });
            return handle_tram(
                prompt,
                cli.knowledge_base,
                cli.probability,
                cli.verbose,
                cli.iterations,
                cli.inference_depth,
                cli.cf_threshold,
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
        } => {
            handle_tram(
                prompt,
                knowledge_base,
                probability,
                cli.verbose,
                iterations,
                inference_depth,
                cf_threshold,
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
            let decoded = crate::encode::decode_from_parts(&equation, length, &residuals)?;
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
                Some(path) => Lexicon::load_from(std::path::Path::new(&path)).ok(),
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
            let aggregator = crate::net::SearchAggregator::new().with_region(&region);
            divider("DuckDuckGo Results");

            if let Err(e) = aggregator.search_and_display(&prompt, limit).await {
                error!("  ⚠️  Search display failed: {}", e);
            }

            let corpus = match aggregator.get_response(&prompt).await {
                Ok(resp) => crate::net::corpus_from_response(&resp),
                Err(_) => String::new(),
            };

            let final_corpus = if corpus.trim().is_empty() {
                let lite_results = aggregator.fetch(&prompt, limit).await.unwrap_or_default();
                let quality = crate::net::corpus_from_results(&lite_results);
                if quality.trim().is_empty() {
                    crate::net::corpus_from_results_raw(&lite_results)
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
            let parsed_style = style
                .parse::<crate::imagen::StyleMode>()
                .unwrap_or(crate::imagen::StyleMode::Plasma);
            let params = crate::imagen::ImagenParams {
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
            match crate::imagen::render(&params) {
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
