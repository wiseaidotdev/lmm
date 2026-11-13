// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Configuration
//!
//! Typed configuration for the TRAM reasoning pipeline.

use crate::cognition::thought::classifier::ReasoningMode;

pub const DEFAULT_ITERATIONS: usize = 10;
pub const DEFAULT_INFERENCE_DEPTH: usize = 5;
pub const DEFAULT_CF_THRESHOLD: f64 = 0.25;
pub const DEFAULT_THINK_STEPS: usize = 8;
pub const DEFAULT_THINK_THRESHOLD: f64 = 0.30;
pub const DEFAULT_THINK_PATIENCE: usize = 3;
pub const DEFAULT_MAX_TASKS: usize = 3;
pub const DEFAULT_STOCHASTIC_PROBABILITY: f64 = 0.12;
pub const DEFAULT_CONVERGENCE_THRESHOLD: f64 = 0.35;
pub const DEFAULT_K_PROPORTIONAL: f64 = 1.2;
pub const DEFAULT_K_INTEGRAL: f64 = 0.08;
pub const DEFAULT_THINKLOOP_ITERATIONS_WITH_KB: usize = 6;
pub const DEFAULT_TOP_K_MIN: usize = 6;
pub const DEFAULT_TOP_K_VAR: u64 = 5;
pub const DEFAULT_MIN_RULE_COUNT: usize = 3;
pub const DEFAULT_MIN_EXPERT_WORDS: usize = 25;
pub const DEFAULT_DEDUP_THRESHOLD: f64 = 0.70;
pub const DEFAULT_PROSE_MIN_SENTENCES: usize = 2;
pub const DEFAULT_PROSE_MAX_SENTENCES: usize = 7;
pub const DEFAULT_MIN_KB_WORDS: usize = 15;
pub const DEFAULT_MIN_PROSE_WORDS: usize = 25;
pub const DEFAULT_RULE_SOURCE_LIMIT: usize = 12;
pub const DEFAULT_USE_ASSEMBLY: bool = true;
pub const DEFAULT_USE_SUMMARIZER_FALLBACK: bool = true;

/// Configuration for the TRAM reasoning pipeline.
#[derive(Debug, Clone)]
pub struct TramConfig {
    pub iterations: usize,
    pub inference_depth: usize,
    pub cf_threshold: f64,
    pub think: bool,
    pub think_steps: usize,
    pub think_threshold: f64,
    pub think_patience: usize,
    pub max_tasks: usize,
    pub probability: Option<f64>,
    pub no_frame: bool,
    pub no_web: bool,
    pub deep: bool,
    pub convergence_threshold: f64,
    pub k_proportional: f64,
    pub k_integral: f64,
    pub top_k_min: usize,
    pub top_k_var: u64,
    pub min_rule_count: usize,
    pub min_expert_words: usize,
    pub dedup_threshold: f64,
    pub prose_min_sentences: usize,
    pub prose_max_sentences: usize,
    pub min_kb_words: usize,
    pub min_prose_words: usize,
    pub rule_source_limit: usize,
    pub use_assembly: bool,
    pub use_summarizer_fallback: bool,
}

#[derive(Debug, Clone, Default)]
pub struct WebFetchResult {
    pub total_chunks_ingested: usize,
    pub prose_chunks_ingested: usize,
    pub lite_snippets_found: usize,
    pub instant_answer_query: Option<String>,
    pub fetch_queries: Vec<String>,
    pub unscraped_urls: Vec<String>,
    pub all_scraped_urls: Vec<String>,
}

impl Default for TramConfig {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_ITERATIONS,
            inference_depth: DEFAULT_INFERENCE_DEPTH,
            cf_threshold: DEFAULT_CF_THRESHOLD,
            think: false,
            think_steps: DEFAULT_THINK_STEPS,
            think_threshold: DEFAULT_THINK_THRESHOLD,
            think_patience: DEFAULT_THINK_PATIENCE,
            max_tasks: DEFAULT_MAX_TASKS,
            probability: None,
            no_frame: false,
            no_web: false,
            deep: false,
            convergence_threshold: DEFAULT_CONVERGENCE_THRESHOLD,
            k_proportional: DEFAULT_K_PROPORTIONAL,
            k_integral: DEFAULT_K_INTEGRAL,
            top_k_min: DEFAULT_TOP_K_MIN,
            top_k_var: DEFAULT_TOP_K_VAR,
            min_rule_count: DEFAULT_MIN_RULE_COUNT,
            min_expert_words: DEFAULT_MIN_EXPERT_WORDS,
            dedup_threshold: DEFAULT_DEDUP_THRESHOLD,
            prose_min_sentences: DEFAULT_PROSE_MIN_SENTENCES,
            prose_max_sentences: DEFAULT_PROSE_MAX_SENTENCES,
            min_kb_words: DEFAULT_MIN_KB_WORDS,
            min_prose_words: DEFAULT_MIN_PROSE_WORDS,
            rule_source_limit: DEFAULT_RULE_SOURCE_LIMIT,
            use_assembly: DEFAULT_USE_ASSEMBLY,
            use_summarizer_fallback: DEFAULT_USE_SUMMARIZER_FALLBACK,
        }
    }
}

impl TramConfig {
    pub fn builder() -> TramConfigBuilder {
        TramConfigBuilder::default()
    }

    pub fn effective_probability(&self) -> f64 {
        self.probability
            .unwrap_or(DEFAULT_STOCHASTIC_PROBABILITY)
            .clamp(0.0, 1.0)
    }

    pub fn thinkloop_iterations(&self, has_kb: bool) -> usize {
        if has_kb {
            self.iterations.min(DEFAULT_THINKLOOP_ITERATIONS_WITH_KB)
        } else {
            self.iterations
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TramConfigBuilder {
    iterations: Option<usize>,
    inference_depth: Option<usize>,
    cf_threshold: Option<f64>,
    think: bool,
    think_steps: Option<usize>,
    think_threshold: Option<f64>,
    think_patience: Option<usize>,
    max_tasks: Option<usize>,
    probability: Option<f64>,
    no_frame: bool,
    no_web: bool,
    deep: bool,
    convergence_threshold: Option<f64>,
    k_proportional: Option<f64>,
    k_integral: Option<f64>,
    top_k_min: Option<usize>,
    top_k_var: Option<u64>,
    min_rule_count: Option<usize>,
    min_expert_words: Option<usize>,
    dedup_threshold: Option<f64>,
    prose_min_sentences: Option<usize>,
    prose_max_sentences: Option<usize>,
    min_kb_words: Option<usize>,
    min_prose_words: Option<usize>,
    rule_source_limit: Option<usize>,
    use_assembly: bool,
    use_summarizer_fallback: bool,
}

impl TramConfigBuilder {
    pub fn iterations(mut self, value: usize) -> Self {
        self.iterations = Some(value);
        self
    }

    pub fn inference_depth(mut self, value: usize) -> Self {
        self.inference_depth = Some(value);
        self
    }

    pub fn cf_threshold(mut self, value: f64) -> Self {
        self.cf_threshold = Some(value);
        self
    }

    pub fn think(mut self, value: bool) -> Self {
        self.think = value;
        self
    }

    pub fn think_steps(mut self, value: usize) -> Self {
        self.think_steps = Some(value);
        self
    }

    pub fn think_threshold(mut self, value: f64) -> Self {
        self.think_threshold = Some(value);
        self
    }

    pub fn think_patience(mut self, value: usize) -> Self {
        self.think_patience = Some(value);
        self
    }

    pub fn max_tasks(mut self, value: usize) -> Self {
        self.max_tasks = Some(value);
        self
    }

    pub fn probability(mut self, value: f64) -> Self {
        self.probability = Some(value);
        self
    }

    pub fn no_frame(mut self, value: bool) -> Self {
        self.no_frame = value;
        self
    }

    pub fn no_web(mut self, value: bool) -> Self {
        self.no_web = value;
        self
    }

    pub fn deep(mut self, value: bool) -> Self {
        self.deep = value;
        self
    }

    pub fn convergence_threshold(mut self, value: f64) -> Self {
        self.convergence_threshold = Some(value);
        self
    }

    pub fn k_proportional(mut self, value: f64) -> Self {
        self.k_proportional = Some(value);
        self
    }

    pub fn k_integral(mut self, value: f64) -> Self {
        self.k_integral = Some(value);
        self
    }

    pub fn top_k_var(mut self, value: u64) -> Self {
        self.top_k_var = Some(value);
        self
    }

    pub fn min_kb_words(mut self, value: usize) -> Self {
        self.min_kb_words = Some(value);
        self
    }

    pub fn min_prose_words(mut self, value: usize) -> Self {
        self.min_prose_words = Some(value);
        self
    }

    pub fn rule_source_limit(mut self, value: usize) -> Self {
        self.rule_source_limit = Some(value);
        self
    }

    pub fn use_assembly(mut self, value: bool) -> Self {
        self.use_assembly = value;
        self
    }

    pub fn use_summarizer_fallback(mut self, value: bool) -> Self {
        self.use_summarizer_fallback = value;
        self
    }

    pub fn build(self) -> TramConfig {
        TramConfig {
            iterations: self.iterations.unwrap_or(DEFAULT_ITERATIONS),
            inference_depth: self.inference_depth.unwrap_or(DEFAULT_INFERENCE_DEPTH),
            cf_threshold: self.cf_threshold.unwrap_or(DEFAULT_CF_THRESHOLD),
            think: self.think,
            think_steps: self.think_steps.unwrap_or(DEFAULT_THINK_STEPS),
            think_threshold: self.think_threshold.unwrap_or(DEFAULT_THINK_THRESHOLD),
            think_patience: self.think_patience.unwrap_or(DEFAULT_THINK_PATIENCE),
            max_tasks: self.max_tasks.unwrap_or(DEFAULT_MAX_TASKS),
            probability: self.probability,
            no_frame: self.no_frame,
            no_web: self.no_web,
            deep: self.deep,
            convergence_threshold: self
                .convergence_threshold
                .unwrap_or(DEFAULT_CONVERGENCE_THRESHOLD),
            k_proportional: self.k_proportional.unwrap_or(DEFAULT_K_PROPORTIONAL),
            k_integral: self.k_integral.unwrap_or(DEFAULT_K_INTEGRAL),
            top_k_min: self.top_k_min.unwrap_or(DEFAULT_TOP_K_MIN),
            top_k_var: self.top_k_var.unwrap_or(DEFAULT_TOP_K_VAR),
            min_rule_count: self.min_rule_count.unwrap_or(DEFAULT_MIN_RULE_COUNT),
            min_expert_words: self.min_expert_words.unwrap_or(DEFAULT_MIN_EXPERT_WORDS),
            dedup_threshold: self.dedup_threshold.unwrap_or(DEFAULT_DEDUP_THRESHOLD),
            prose_min_sentences: self
                .prose_min_sentences
                .unwrap_or(DEFAULT_PROSE_MIN_SENTENCES),
            prose_max_sentences: self
                .prose_max_sentences
                .unwrap_or(DEFAULT_PROSE_MAX_SENTENCES),
            min_kb_words: self.min_kb_words.unwrap_or(DEFAULT_MIN_KB_WORDS),
            min_prose_words: self.min_prose_words.unwrap_or(DEFAULT_MIN_PROSE_WORDS),
            rule_source_limit: self.rule_source_limit.unwrap_or(DEFAULT_RULE_SOURCE_LIMIT),
            use_assembly: self.use_assembly,
            use_summarizer_fallback: self.use_summarizer_fallback,
        }
    }
}

/// Result of a TRAM pipeline execution.
#[derive(Debug, Clone)]
pub struct TramResult {
    pub final_text: String,
    pub web_fetch_result: WebFetchResult,
    pub ate_result: Option<AteResult>,
    pub thinkloop_result: ThinkLoopResult,
    pub kb_retrieval_result: KbRetrievalResult,
    pub expert_result: Option<ExpertResult>,
    pub stochastic_result: StochasticResult,
}

#[derive(Debug, Clone)]
pub struct AteResult {
    pub mode: ReasoningMode,
    pub steps: usize,
    pub confidence: f64,
    pub factual_insight: String,
    pub prioritized_query: String,
}

#[derive(Debug, Clone)]
pub struct ThinkLoopResult {
    pub converged: bool,
    pub steps: usize,
    pub final_error: f64,
    pub cold_corpus: String,
    pub chunks_added_to_kb: usize,
}

#[derive(Debug, Clone)]
pub struct KbRetrievalResult {
    pub answer_found: bool,
    pub answer_quality: bool,
    pub intent_relevant: bool,
    pub answer_source: AnswerSource,
    pub topic: String,
    pub kb_query: String,
    pub top_k: usize,
    pub kb_chunks: Vec<String>,
    pub answer_text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnswerSource {
    KnowledgeBase,
    SymbolicFallback,
    ColdCorpusSummary,
}

#[derive(Debug, Clone)]
pub struct ExpertResult {
    pub conclusion: String,
    pub certainty: f64,
    pub depth: usize,
    pub rule_count: usize,
    pub trace: String,
    pub prose: String,
}

#[derive(Debug, Clone)]
pub struct StochasticResult {
    pub text: String,
    pub substitutions: usize,
    pub language: String,
    pub probability: f64,
}
