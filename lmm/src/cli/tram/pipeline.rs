// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Pipeline CLI Adapter
//!
//! CLI wrapper for the core TRAM pipeline, handling display and logging.

use crate::cli::display;
use crate::cognition::knowledge::KnowledgeIndex;
use crate::tram::{
    TramConfig, TramPipeline, prefetch_web as tram_prefetch_web,
    prefetch_web_for_query as tram_prefetch_web_for_query,
};
use tracing::info;

pub async fn prefetch_web(
    prompt: &str,
    index: &mut KnowledgeIndex,
    kb_provided: bool,
    deep: bool,
    verbose: bool,
) {
    display::divider("Web Knowledge");
    let result = tram_prefetch_web(prompt, index, kb_provided, deep, verbose).await;

    if result.total_chunks_ingested > 0 {
        if let Some(ia_query) = &result.instant_answer_query {
            if result.prose_chunks_ingested > 0 {
                info!(
                    "  🌐 Instant Answer [{:?}]: {} chunk(s) ingested ({} prose)",
                    ia_query, result.total_chunks_ingested, result.prose_chunks_ingested
                );
            } else {
                info!(
                    "  🌐 Instant Answer [{:?}]: {} fragment(s), augmenting with Lite search",
                    ia_query, result.total_chunks_ingested
                );
            }
        }
        if result.lite_snippets_found > 0 {
            info!(
                "  🌐 Lite search: {} snippet chunk(s) ingested",
                result.lite_snippets_found
            );
        }
    } else if verbose {
        info!("  🌐 Web Search: no results");
    }

    if deep && verbose {
        info!("  🔍 Deep Search: enabled");
    }
}

pub async fn prefetch_web_for_query(query: &str, index: &mut KnowledgeIndex, verbose: bool) {
    let result = tram_prefetch_web_for_query(query, index, verbose).await;

    if result.total_chunks_ingested > 0 && verbose {
        info!(
            "  🌐 Intent query [{:?}]: {} chunk(s) ingested",
            query, result.total_chunks_ingested
        );
    }
}

pub async fn run_tram_pipeline(
    prompt: String,
    config: TramConfig,
    index: &mut KnowledgeIndex,
    verbose: bool,
) -> anyhow::Result<String> {
    display::banner("TRAM · Text Reasoning and Applied Mathematics", "🧮");
    info!("  ❓ Prompt    : {:?}", prompt);
    info!("  🔁 Iterations: {}", config.iterations);
    info!(
        "  🔍 Inference : depth={} CF≥{:.2}",
        config.inference_depth, config.cf_threshold
    );

    let pipeline = TramPipeline::new(config);
    let result = pipeline.run(&prompt, index).await?;

    if verbose {
        if let Some(ref ate) = result.ate_result {
            display::divider("Adaptive Thought Engine");
            info!(
                "  🧠 ATE: mode={} steps={} confidence={:.3}",
                ate.mode.label(),
                ate.steps,
                ate.confidence
            );
        }

        display::divider("ThinkLoop Reasoning");
        info!(
            "  🔄 ThinkLoop: converged={} steps={} error={:.3}",
            result.thinkloop_result.converged,
            result.thinkloop_result.steps,
            result.thinkloop_result.final_error
        );

        display::divider("Knowledge Retrieval");
        if result.kb_retrieval_result.answer_found {
            info!(
                "  ✅ KB answer found ({} chars)",
                result.kb_retrieval_result.answer_text.len()
            );
        } else {
            info!("  🔀 KB answer not found. Building symbolic response");
        }

        display::divider("Expert Inference");
        if let Some(ref expert) = result.expert_result {
            info!(
                "  🧠 Inference: conclusion={:?} CF={:.2} depth={} rules={}",
                expert.conclusion, expert.certainty, expert.depth, expert.rule_count
            );
            info!("  📋 Explanation: {} steps", expert.trace.len());
        } else {
            info!(
                "  ℹ️  Expert inference: no conclusion (threshold={:.2})",
                pipeline.config().cf_threshold
            );
        }

        display::divider("Stochastic Enhancement");
        info!(
            "  🎲 substitutions={} language={:?} prob={:.2}",
            result.stochastic_result.substitutions,
            result.stochastic_result.language,
            result.stochastic_result.probability
        );

        display::divider("TRAM Response");
        info!("  {}", result.final_text);
    }

    Ok(result.final_text)
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
