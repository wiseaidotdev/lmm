// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Loop Orchestrator
//!
//! Implements `--loop N` recursive multi-pass pipeline execution using
//! Bayesian belief propagation (Pearl, 1988) to track and improve response
//! quality across passes.
//!
//! Each pass:
//! 1. Runs the full TRAM pipeline (knowledge → thinking → retrieval → assembly).
//! 2. Builds a [`BeliefNet`] from top KB chunks and propagates prompt evidence.
//! 3. Fuses the Bayesian posterior with the DS belief interval midpoint.
//! 4. Records quality and exits early when the improvement delta falls below the
//!    configured threshold.
//!
//! All format strings, caps, and weights come from [`crate::constants::tram_loop`].
//!
//! ## See Also
//!
//! - [Bayesian network - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_network)
//! - [Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.](https://dl.acm.org/doi/book/10.5555/534975)
//! - [Belief propagation - Wikipedia](https://en.wikipedia.org/wiki/Belief_propagation)
//! - [Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations. ICML.](https://dl.acm.org/doi/10.5555/645528.657613)

#[cfg(feature = "cli")]
use {
    crate::{
        app::tram::{assembly, knowledge, retrieval, rewards, thinking},
        cli::{
            display::divider,
            tram::{
                query::{extract_creative_subject, extract_topic},
                response::derive_kb_query,
            },
        },
        cognition::{
            bayesian::BeliefNet, task_decomposer::TaskDecomposer, thought::belief::BeliefInterval,
        },
        constants::tram_loop::{BAYESIAN_FUSION_WEIGHT, LOOP_MAX_COUNT, LOOP_PASS_DIVIDER_LABEL},
    },
    std::path::PathBuf,
    tracing::info,
};

/// The accumulated state carried across loop passes.
#[cfg(feature = "cli")]
pub struct LoopState {
    /// Best answer text produced so far.
    pub answer: String,
    /// Quality score of `answer` in `[0.0, 1.0]`.
    pub quality: f64,
    /// Fused DS belief interval after the last pass.
    pub belief: BeliefInterval,
    /// Pass number on which this state was produced.
    pub pass: usize,
}

/// Configuration for the loop orchestrator.
#[cfg(feature = "cli")]
pub struct LoopConfig {
    /// Number of pipeline passes requested (clamped to [`LOOP_MAX_COUNT`]).
    pub loop_count: usize,
    /// Minimum quality improvement per pass required to keep iterating.
    pub loop_threshold: f64,
    /// Whether to emit per-pass diagnostic logs.
    pub verbose: bool,
}

/// Runs the TRAM pipeline repeatedly for `loop_config.loop_count` passes,
/// propagating Bayesian beliefs across iterations.
///
/// # Arguments
///
/// * `prompt`          - User question.
/// * `knowledge_base`  - Optional path to a local KB.
/// * `probability`     - Stochastic synonym probability override.
/// * `iterations`      - ThinkLoop iteration budget.
/// * `inference_depth` - Expert backward-chaining depth.
/// * `cf_threshold`    - Expert certainty-factor threshold.
/// * `think`           - Activate Adaptive Thought Engine.
/// * `think_steps`     - ATE thought-step budget.
/// * `think_threshold` - ATE early-stop belief threshold.
/// * `think_patience`  - ATE stagnation patience.
/// * `max_tasks`       - Task-decomposer sub-task limit.
/// * `no_frame`        - Skip response framing.
/// * `deep`            - Activate deep-scrape mode.
/// * `loop_config`     - Loop control parameters.
/// * `skill_dir`       - Persistent skill storage directory.
///
/// # Returns
///
/// The best [`LoopState`] achieved across all passes.
///
/// # Time Complexity
///
/// O(loop_count × single_pass_cost)
#[cfg(feature = "cli")]
#[allow(clippy::too_many_arguments)]
pub async fn run_loop(
    prompt: &str,
    knowledge_base: Option<String>,
    probability: Option<f64>,
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
    loop_config: LoopConfig,
    skill_dir: PathBuf,
) -> anyhow::Result<LoopState> {
    let pass_cap = loop_config.loop_count.clamp(1, LOOP_MAX_COUNT);

    let mut state = LoopState {
        answer: String::new(),
        quality: 0.0,
        belief: BeliefInterval::ignorant(),
        pass: 0,
    };

    let prompt_tokens: Vec<&str> = prompt.split_whitespace().collect();

    for pass in 1..=pass_cap {
        divider(&format!(
            "{} {}/{}",
            LOOP_PASS_DIVIDER_LABEL, pass, pass_cap
        ));

        divider("Web Knowledge");
        let mut ks = knowledge::run_knowledge_stage(
            prompt,
            knowledge_base.clone(),
            deep,
            loop_config.verbose,
            max_tasks,
            skill_dir.clone(),
        )
        .await;

        let ts = thinking::run_thinking_stage(thinking::ThinkingConfig {
            index: &mut ks.index,
            prompt,
            think,
            think_steps,
            think_threshold,
            think_patience,
            iterations,
            verbose: loop_config.verbose,
        })
        .await;

        let top_chunks: Vec<String> = ks
            .index
            .query(&extract_topic(prompt), 12)
            .into_iter()
            .map(|c| c.text.clone())
            .collect();

        let chunk_refs: Vec<&str> = top_chunks.iter().map(|s| s.as_str()).collect();
        let mut belief_net = BeliefNet::from_chunks(&chunk_refs, &extract_topic(prompt));
        let posterior = belief_net.propagate(&prompt_tokens);

        let fused_mid = BAYESIAN_FUSION_WEIGHT * posterior
            + (1.0 - BAYESIAN_FUSION_WEIGHT) * state.belief.mid();
        let new_belief = BeliefInterval::new(fused_mid * 0.9, (fused_mid + 0.1).min(1.0));

        let topic = if TaskDecomposer::is_creative_task(prompt) {
            extract_creative_subject(prompt)
        } else {
            extract_topic(prompt)
        };
        let kb_query = derive_kb_query(prompt, ks.prompt_mode, &topic);

        divider("Knowledge Retrieval");
        let rs = retrieval::run_retrieval_stage(retrieval::RetrievalConfig {
            index: &ks.index,
            prompt,
            topic: &topic,
            prompt_mode: ks.prompt_mode,
            run_seed: ts.run_seed,
            think,
            deep,
            kb_query: &kb_query,
        });

        let all_scraped_urls = ks.web_result.successful_urls.clone();
        let answer_text = assembly::run_assembly_stage(
            rs.kb_chunks,
            rs.combined_answer,
            &ts.cold_corpus,
            prompt,
            &topic,
            ks.prompt_mode,
            ts.run_seed,
            think,
            deep,
            no_frame,
            loop_config.verbose,
            probability,
            inference_depth,
            cf_threshold,
            &kb_query,
            &all_scraped_urls,
        );

        let new_quality = rewards::compute_response_reward(&answer_text, prompt);
        let delta = new_quality - state.quality;

        if loop_config.verbose {
            info!(
                "  🔁 Pass {}/{}: quality={:.3} delta={:+.3} belief_mid={:.3}",
                pass,
                pass_cap,
                new_quality,
                delta,
                new_belief.mid()
            );
        }

        let episode_reward = new_quality;
        let topic_for_skill = topic.clone();
        ks.learning_engine.end_of_episode(
            &ts.cold_store,
            &mut ks.index,
            &topic_for_skill,
            episode_reward,
        );
        let _ = ks.skill_registry.flush();
        let _ = ks.learning_engine.save(&skill_dir);

        if new_quality > state.quality || pass == 1 {
            state.answer = answer_text;
            state.quality = new_quality;
            state.belief = new_belief;
            state.pass = pass;
        }

        if pass > 1 && delta < loop_config.loop_threshold {
            if loop_config.verbose {
                info!(
                    "  ✅ Loop converged at pass {} (delta={:.3} < threshold={:.3})",
                    pass, delta, loop_config.loop_threshold
                );
            }
            break;
        }
    }

    if loop_config.verbose && state.pass == pass_cap {
        info!(
            "  🔚 Loop exhausted {} passes (final quality={:.3})",
            pass_cap, state.quality
        );
    }

    Ok(state)
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
