// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

use crate::types::{CodeBlock, PipelineStage, SearchLink, UploadedFile};
use lmm::cognition::knowledge::KnowledgeIndex;
use lmm::stochastic::StochasticEnhancer;
use lmm::text::TextSummarizer;
use lmm::tram::{TramConfig, TramPipeline};
use yew::Callback;

fn parse_code_blocks(content: &str) -> (String, Vec<CodeBlock>) {
    let mut blocks = Vec::new();
    let mut prose_parts = Vec::new();
    let mut remaining = content;

    while !remaining.is_empty() {
        if let Some(fence_start) = remaining.find("```") {
            let before = &remaining[..fence_start];
            if !before.trim().is_empty() {
                prose_parts.push(before.trim().to_string());
            }

            let after_fence = &remaining[fence_start + 3..];
            let lang_end = after_fence
                .find(|c: char| c == '\n' || c == '\r')
                .unwrap_or(0);
            let language = after_fence[..lang_end].trim().to_string();
            let code_start = lang_end + 1;

            if let Some(end_fence) = after_fence[code_start..].find("```") {
                let code = after_fence[code_start..code_start + end_fence]
                    .trim()
                    .to_string();
                if !code.is_empty() {
                    blocks.push(CodeBlock { language, code });
                }
                remaining = &after_fence[code_start + end_fence + 3..];
            } else {
                prose_parts.push(after_fence.trim().to_string());
                break;
            }
        } else {
            if !remaining.trim().is_empty() {
                prose_parts.push(remaining.trim().to_string());
            }
            break;
        }
    }

    (prose_parts.join("\n\n"), blocks)
}

fn links_from_urls(urls: &[String]) -> Vec<SearchLink> {
    urls.iter()
        .filter(|u| !u.is_empty())
        .map(|url| {
            let title = url
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .split('/')
                .take(3)
                .collect::<Vec<_>>()
                .join("/");
            SearchLink {
                title,
                url: url.clone(),
            }
        })
        .collect()
}

async fn run_pipeline_with_progress(
    prompt: &str,
    config: TramConfig,
    index: &mut KnowledgeIndex,
    on_progress: &Callback<PipelineStage>,
) -> (String, Vec<CodeBlock>, Vec<SearchLink>) {
    on_progress.emit(PipelineStage::SearchingWeb);

    let pipeline = TramPipeline::new(config);

    on_progress.emit(PipelineStage::AdaptiveThoughtEngine);

    let result = match pipeline.run(prompt, index).await {
        Ok(r) => r,
        Err(_) => {
            return (
                "An internal error occurred in the TRAM pipeline.".to_string(),
                vec![],
                vec![],
            );
        }
    };

    on_progress.emit(PipelineStage::Composing);

    let links = links_from_urls(&result.web_fetch_result.all_scraped_urls);
    let (prose, code_blocks) = parse_code_blocks(&result.final_text);

    (prose, code_blocks, links)
}

pub async fn tram_query(
    prompt: &str,
    think: bool,
    deep: bool,
    iterations: usize,
    inference_depth: usize,
    cf_threshold: f64,
    stochastic: bool,
    probability: f64,
    on_progress: Callback<PipelineStage>,
) -> (String, Vec<CodeBlock>, Vec<SearchLink>) {
    let config = TramConfig::builder()
        .think(think)
        .deep(deep)
        .iterations(iterations)
        .inference_depth(inference_depth)
        .cf_threshold(cf_threshold)
        .probability(if stochastic { probability } else { 0.0 })
        .build();

    let mut index = KnowledgeIndex::new();
    run_pipeline_with_progress(prompt, config, &mut index, &on_progress).await
}

pub async fn tram_query_with_knowledge(
    prompt: &str,
    files: &[UploadedFile],
    think: bool,
    deep: bool,
    iterations: usize,
    inference_depth: usize,
    cf_threshold: f64,
    stochastic: bool,
    probability: f64,
    on_progress: Callback<PipelineStage>,
) -> (String, Vec<CodeBlock>, Vec<SearchLink>) {
    on_progress.emit(PipelineStage::KnowledgeRetrieval);

    let mut index = KnowledgeIndex::new();
    for file in files {
        index.ingest_text(&file.name, &file.text);
        index.ingest_paragraphs(&file.name, &file.text);
    }

    let config = TramConfig::builder()
        .think(think)
        .deep(deep)
        .iterations(iterations)
        .inference_depth(inference_depth)
        .cf_threshold(cf_threshold)
        .probability(if stochastic { probability } else { 0.0 })
        .build();

    run_pipeline_with_progress(prompt, config, &mut index, &on_progress).await
}

pub fn summarize(text: &str, count: usize, stochastic: bool, prob: f64) -> String {
    let enhance = |s: &str| -> String {
        if stochastic {
            StochasticEnhancer::new(prob).enhance(s)
        } else {
            s.to_string()
        }
    };
    match TextSummarizer::new(count.max(1), 50, 5).summarize(text) {
        Ok(sentences) => sentences
            .iter()
            .map(|s| enhance(s))
            .collect::<Vec<_>>()
            .join(" "),
        Err(_) => {
            "Input needs complete sentences (ending with . ! ?) of at least 5 words.".to_string()
        }
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.
