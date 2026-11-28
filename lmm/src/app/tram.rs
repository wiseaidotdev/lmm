// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Pipeline Sub-stages
//!
//! This module exposes [`handle_tram`], the top-level async entry-point for
//! the TRAM reasoning pipeline.  The pipeline is split across four stage
//! sub-modules:
//!
//! | Module              | Responsibility                                        |
//! |---------------------|-------------------------------------------------------|
//! | [`knowledge`]       | KB loading, web pre-fetch, skill registry, task decomp |
//! | [`thinking`]        | Adaptive Thought Engine + ThinkLoop PID reasoner      |
//! | [`retrieval`]       | KB answer retrieval + deep-scrape quality filter      |
//! | [`assembly`]        | Expert inference, stochastic, framing, URL footer     |
//! | [`rewards`]         | Episode reward scoring + evaluative point synthesis   |

pub mod assembly;
pub mod knowledge;
pub mod loop_orchestrator;
pub mod retrieval;
pub mod rewards;
pub mod thinking;

#[cfg(feature = "cli")]
use {
    crate::{
        cli::{
            display::{banner, divider},
            tram::{
                query::{extract_creative_subject, extract_topic},
                response::derive_kb_query,
            },
        },
        cognition::task_decomposer::TaskDecomposer,
        constants::skill::SKILL_STORE_DIR,
    },
    std::path::PathBuf,
    tracing::info,
};

/// Executes the TRAM (Text Reasoning and Applied Mathematics) pipeline.
///
/// TRAM is the primary reasoning mode of the `lmm` binary, a CPU-only,
/// training-free alternative to GPT-style language models.  It orchestrates
/// five pipeline stages:
///
/// ```text
/// prompt
///   ├─► knowledge  - KB load, web fetch, skill registry, task decomp
///   ├─► thinking   - ATE (optional) + ThinkLoop PID reasoner
///   ├─► retrieval  - KB answer, analogical / evaluative / deep paths
///   ├─► assembly   - expert inference, stochastic, framing, URL footer
///   └─► rewards    - episode reward → LearningEngine
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
/// * `think`            - Whether to run the Adaptive Thought Engine pre-pass.
/// * `think_steps`      - Maximum thought steps for ATE.
/// * `think_threshold`  - DS-belief midpoint early-stop threshold for ATE [0, 1].
/// * `think_patience`   - Consecutive non-improving steps before ATE flags stagnation.
/// * `max_tasks`        - Upper bound on task-decomposer sub-tasks.
/// * `no_frame`         - Skip response framing when `true`.
/// * `deep`             - Activate deep-scrape web enrichment mode.
/// * `loop_count`       - Number of recursive belief-propagation passes (1 = single pass).
/// * `loop_threshold`   - Minimum quality improvement per pass for early-exit.
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
    loop_count: usize,
    loop_threshold: f64,
) -> anyhow::Result<()> {
    banner("TRAM · Text Reasoning and Applied Mathematics", "🧮");
    info!("  ❓ Prompt    : {:?}", prompt);
    info!("  🔁 Iterations: {}", iterations);
    info!(
        "  🔍 Inference : depth={} CF≥{:.2}",
        inference_depth, cf_threshold
    );

    let skill_dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(SKILL_STORE_DIR);

    if loop_count > 1 {
        let lc = loop_orchestrator::LoopConfig {
            loop_count,
            loop_threshold,
            verbose,
        };
        let final_state = loop_orchestrator::run_loop(
            &prompt,
            knowledge_base,
            probability,
            iterations,
            inference_depth,
            cf_threshold,
            think,
            think_steps,
            think_threshold,
            think_patience,
            max_tasks,
            no_frame,
            deep,
            lc,
            skill_dir,
        )
        .await?;
        divider("TRAM Response");
        if verbose {
            info!("  {}", final_state.answer);
        } else {
            info!("{}", final_state.answer);
        }
        return Ok(());
    }

    divider("Web Knowledge");

    let mut ks = knowledge::run_knowledge_stage(
        &prompt,
        knowledge_base,
        deep,
        verbose,
        max_tasks,
        skill_dir,
    )
    .await;

    let ts = thinking::run_thinking_stage(thinking::ThinkingConfig {
        index: &mut ks.index,
        prompt: &prompt,
        think,
        think_steps,
        think_threshold,
        think_patience,
        iterations,
        verbose,
    })
    .await;

    divider("Knowledge Retrieval");

    let topic = if TaskDecomposer::is_creative_task(&prompt) {
        extract_creative_subject(&prompt)
    } else {
        extract_topic(&prompt)
    };
    let kb_query = derive_kb_query(&prompt, ks.prompt_mode, &topic);

    let rs = retrieval::run_retrieval_stage(retrieval::RetrievalConfig {
        index: &ks.index,
        prompt: &prompt,
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
        &prompt,
        &topic,
        ks.prompt_mode,
        ts.run_seed,
        think,
        deep,
        no_frame,
        verbose,
        probability,
        inference_depth,
        cf_threshold,
        &kb_query,
        &all_scraped_urls,
    );

    if verbose {
        divider("TRAM Response");
        info!("  {}", answer_text);
    } else {
        info!("{}", answer_text);
    }

    let episode_reward = rewards::compute_response_reward(&answer_text, &prompt);
    let topic_for_skill = topic.clone();
    ks.learning_engine.end_of_episode(
        &ts.cold_store,
        &mut ks.index,
        &topic_for_skill,
        episode_reward,
    );
    if let Err(e) = ks.skill_registry.flush()
        && verbose
    {
        info!("  ⚠️  Skill registry flush failed: {e}");
    }
    if let Err(e) = ks.learning_engine.save(&ks.skill_dir)
        && verbose
    {
        info!("  ⚠️  Learning engine save failed: {e}");
    }

    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
