// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Thinking Stage
//!
//! Runs the Adaptive Thought Engine (optional) and the ThinkLoop PID reasoner.
//! Outputs a canonical goal query, the cold-memory corpus, and a per-run seed
//! derived from the current clock nanoseconds.

#[cfg(feature = "cli")]
use {
    crate::{
        cli::{display::divider, tram::query::extract_topic},
        cognition::{
            knowledge::KnowledgeIndex, r#loop::ThinkLoop, memory::ColdStore, search::SearchOracle,
            thought::AdaptiveThoughtEngine,
        },
    },
    std::time::{SystemTime, UNIX_EPOCH},
    tracing::info,
};

/// Data produced by the thinking stage and consumed by subsequent stages.
#[cfg(feature = "cli")]
pub struct ThinkingStageOutput {
    /// The goal query forwarded to the ThinkLoop (may equal the original prompt).
    pub ate_goal: String,
    /// Cold-memory corpus accumulated by the ThinkLoop (joined sentences).
    pub cold_corpus: String,
    /// Cold-memory store, passed to `LearningEngine::end_of_episode`.
    pub cold_store: ColdStore,
    /// Per-run seed derived from clock sub-second nanoseconds, used for
    /// deterministic variation in phrase selection and KB top-k.
    pub run_seed: u64,
}

/// Configuration for the thinking stage.
#[cfg(feature = "cli")]
pub struct ThinkingConfig<'a> {
    /// Mutable knowledge index; ATE insights are ingested here.
    pub index: &'a mut KnowledgeIndex,
    /// Original user prompt; used as the goal when `think` is `false`.
    pub prompt: &'a str,
    /// Whether to run the ATE pre-pass.
    pub think: bool,
    /// Maximum ATE thought steps.
    pub think_steps: usize,
    /// DS-belief midpoint early-stop threshold for ATE [0, 1].
    pub think_threshold: f64,
    /// Consecutive non-improving steps before ATE flags stagnation.
    pub think_patience: usize,
    /// Maximum ThinkLoop iterations.
    pub iterations: usize,
    /// Emit per-thought diagnostic logs when `true`.
    pub verbose: bool,
}

/// Runs the thinking stage.
///
/// 1. Optionally executes the Adaptive Thought Engine (ATE) to produce a
///    refined goal query and ingest factual insights into `index`.
/// 2. Always runs the ThinkLoop PID reasoner on the goal query.
/// 3. Flushes ThinkLoop cold-memory entries back into `index`.
///
/// # Returns
///
/// A [`ThinkingStageOutput`] ready to pass to the retrieval stage.
///
/// # Time Complexity
///
/// O(iterations × |oracle|) for the ThinkLoop; O(think_steps) for the ATE.
#[cfg(feature = "cli")]
pub async fn run_thinking_stage(cfg: ThinkingConfig<'_>) -> ThinkingStageOutput {
    let ThinkingConfig {
        index,
        prompt,
        think,
        think_steps,
        think_threshold,
        think_patience,
        iterations,
        verbose,
    } = cfg;
    let run_seed: u64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;

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
        let web_sources = [
            "web-ia",
            "web-ia-eval",
            "web-ducky",
            "web-lite",
            "web-deep-scrape",
            "skill",
        ];
        for source in &web_sources {
            let chunks: String = index
                .query(source, 5)
                .into_iter()
                .map(|c| c.text.clone())
                .collect::<Vec<_>>()
                .join(" ");
            if !chunks.is_empty() {
                oracle.seed_cache(source, &chunks);
            }
        }
    }

    let ate_goal: String = if think {
        divider("Adaptive Thought Engine");

        let ate = AdaptiveThoughtEngine::new(think_steps, think_threshold, think_patience);
        let thought_result = ate.run(prompt, &mut oracle).await;

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
            .unwrap_or_else(|| prompt.to_string())
    } else {
        prompt.to_string()
    };

    divider("ThinkLoop Reasoning");

    let convergence_threshold = 0.45;
    let think_iterations = if index.is_empty() {
        iterations
    } else {
        iterations.min(6)
    };
    let think_goal = ate_goal.clone();
    let mut lp = ThinkLoop::builder(think_goal)
        .max_iterations(think_iterations)
        .convergence_threshold(convergence_threshold)
        .k_proportional(1.2)
        .k_integral(0.08)
        .build();

    let think_result = lp.run(&mut oracle).await;
    let think_converged =
        think_result.converged && !(index.is_empty() && think_result.final_error >= 0.99);
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

    let mut total_chunks: usize = 0;
    for entry in lp.cold.all() {
        total_chunks += index.ingest_text("think", &entry.content);
    }
    if verbose {
        info!("  💡 Memory → KB: +{} chunk(s)", total_chunks);
    }

    let cold_store = lp.cold;

    ThinkingStageOutput {
        ate_goal,
        cold_corpus,
        cold_store,
        run_seed,
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
