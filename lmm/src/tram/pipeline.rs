// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Pipeline
//!
//! Core orchestration of the TRAM reasoning pipeline for programmatic use.

use crate::cognition::{
    code_responder::CodeResponder,
    knowledge::KnowledgeIndex,
    r#loop::ThinkLoop,
    response_framer::ResponseFramer,
    search::SearchOracle,
    task_decomposer::{TaskDecomposer, TaskKind},
    thought::{
        AdaptiveThoughtEngine,
        classifier::{InputClassifier, ReasoningMode},
    },
};
use crate::error::Result as LmmResult;
use crate::expert::{inference::InferenceEngine, rule::RuleBase};
use crate::generation::{
    dedup::GlobalDeduplicator, filter::SentenceFilter, prose::ProseBuilder, shaper::ParagraphShaper,
};
use crate::stochastic::StochasticEnhancer;
use crate::text::TextSummarizer;
use crate::tram::{
    config::{
        AnswerSource, AteResult, ExpertResult, KbRetrievalResult, StochasticResult,
        ThinkLoopResult, TramConfig, TramResult, WebFetchResult,
    },
    prefetch_web,
    query::extract_topic,
    response::{is_quality_answer, restructure_analogical_response},
    web_fetch::prefetch_web_for_query,
};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "wasm32")]
use js_sys::Date;

const ORACLE_CACHE_SIZE: usize = 5;
const KB_QUERY_TOP_K: usize = 8;

/// Main TRAM pipeline orchestrator.
///
/// Provides a reusable API for running the TRAM reasoning pipeline
/// programmatically without CLI dependencies.
pub struct TramPipeline {
    config: TramConfig,
}

impl TramPipeline {
    pub fn new(config: TramConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &TramConfig {
        &self.config
    }

    pub async fn run(&self, prompt: &str, index: &mut KnowledgeIndex) -> LmmResult<TramResult> {
        let kb_provided = !index.is_empty();
        let web_fetch_result = self.fetch_web_knowledge(prompt, index, kb_provided).await;

        let prompt_mode = InputClassifier::classify(prompt);
        let sub_tasks = TaskDecomposer::decompose(prompt, prompt_mode, self.config.max_tasks);
        if !self.config.no_web {
            for task in &sub_tasks {
                if let TaskKind::WebSearch(q) = &task.kind {
                    prefetch_web_for_query(q, index).await;
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let run_seed: u64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        #[cfg(target_arch = "wasm32")]
        let run_seed: u64 = Date::now() as u64;

        let mut oracle = SearchOracle::new(ORACLE_CACHE_SIZE).no_web(self.config.no_web);

        if !index.is_empty() {
            let topic_query = extract_topic(prompt);
            let kb_corpus: String = index
                .query(&topic_query, KB_QUERY_TOP_K)
                .into_iter()
                .map(|c| c.text.clone())
                .collect::<Vec<_>>()
                .join(" ");
            if !kb_corpus.is_empty() {
                oracle.seed_cache(&topic_query, &kb_corpus);
                oracle.seed_cache(prompt, &kb_corpus);
            }
        }

        let ate_result = if self.config.think && !self.config.no_web {
            Some(self.run_ate(prompt, &mut oracle, index, run_seed).await?)
        } else {
            None
        };

        let ate_goal = ate_result
            .as_ref()
            .map(|r| r.prioritized_query.clone())
            .unwrap_or_else(|| prompt.to_string());

        let thinkloop_result = if self.config.no_web {
            ThinkLoopResult {
                converged: false,
                steps: 0,
                final_error: 1.0,
                cold_corpus: String::new(),
                chunks_added_to_kb: 0,
            }
        } else {
            self.run_thinkloop(&ate_goal, &mut oracle, index, run_seed)
                .await
        };

        let topic = extract_topic(prompt);
        let kb_retrieval_result = self.retrieve_knowledge(
            prompt,
            &topic,
            prompt_mode,
            index,
            &thinkloop_result.cold_corpus,
            run_seed,
        );

        let corpus_for_fallback = thinkloop_result.cold_corpus.clone();

        let expert_result = self.run_expert_inference(
            prompt,
            &kb_retrieval_result.kb_chunks,
            self.config.inference_depth,
            self.config.cf_threshold,
            &kb_retrieval_result.answer_text,
            run_seed,
        );

        let answer_text = self.merge_expert_response(
            kb_retrieval_result.answer_text.clone(),
            &expert_result,
            &kb_retrieval_result.kb_chunks,
            run_seed,
        );

        let processed_text = if self.config.use_summarizer_fallback
            && !answer_text.is_empty()
            && (answer_text.split_whitespace().count() < 20
                || answer_text.contains("offline mode")
                || answer_text.contains("knowledge base may not contain"))
        {
            let text_to_summarize = if !corpus_for_fallback.is_empty() {
                &corpus_for_fallback
            } else {
                &answer_text
            };
            if let Ok(sents) =
                TextSummarizer::new(5, 60, 5).summarize_with_query(text_to_summarize, &topic)
            {
                if !sents.is_empty() {
                    sents.join(" ")
                } else if !corpus_for_fallback.is_empty() {
                    corpus_for_fallback
                } else {
                    answer_text
                }
            } else if !corpus_for_fallback.is_empty() {
                corpus_for_fallback
            } else {
                answer_text
            }
        } else {
            answer_text
        };

        let stochastic_result = self.apply_stochastic_enhancement(&processed_text);

        let final_text = if self.config.no_frame {
            stochastic_result.text.clone()
        } else if TaskDecomposer::is_creative_task(prompt) {
            self.build_creative_response(
                prompt,
                &topic,
                &kb_retrieval_result.kb_chunks,
                &stochastic_result.text,
            )
        } else {
            self.build_framed_response(
                prompt,
                prompt_mode,
                &topic,
                &stochastic_result.text,
                run_seed,
            )
        };

        Ok(TramResult {
            final_text,
            web_fetch_result,
            ate_result,
            thinkloop_result,
            kb_retrieval_result,
            expert_result,
            stochastic_result,
        })
    }

    async fn fetch_web_knowledge(
        &self,
        prompt: &str,
        index: &mut KnowledgeIndex,
        kb_provided: bool,
    ) -> WebFetchResult {
        if self.config.no_web {
            return WebFetchResult::default();
        }

        #[cfg(feature = "net")]
        {
            prefetch_web(prompt, index, kb_provided).await
        }
        #[cfg(not(feature = "net"))]
        {
            let _ = (prompt, index, kb_provided);
            WebFetchResult::default()
        }
    }

    async fn run_ate(
        &self,
        prompt: &str,
        oracle: &mut SearchOracle,
        index: &mut KnowledgeIndex,
        _run_seed: u64,
    ) -> LmmResult<AteResult> {
        let ate = AdaptiveThoughtEngine::new(
            self.config.think_steps,
            self.config.think_threshold,
            self.config.think_patience,
        );
        let thought_result = ate.run(prompt, oracle).await;

        if !thought_result.factual_insight.is_empty() {
            index.ingest_text("ate-insight", &thought_result.factual_insight);
        }

        Ok(AteResult {
            mode: thought_result.chain.mode,
            steps: thought_result.chain.steps.len(),
            confidence: thought_result.confidence,
            factual_insight: thought_result.factual_insight,
            prioritized_query: thought_result
                .prioritized_queries
                .into_iter()
                .next()
                .unwrap_or_else(|| prompt.to_string()),
        })
    }

    async fn run_thinkloop(
        &self,
        goal: &str,
        oracle: &mut SearchOracle,
        index: &mut KnowledgeIndex,
        _run_seed: u64,
    ) -> ThinkLoopResult {
        let has_kb = !index.is_empty();
        let think_iterations = self.config.thinkloop_iterations(has_kb);

        let mut lp = ThinkLoop::builder(goal)
            .max_iterations(think_iterations)
            .convergence_threshold(self.config.convergence_threshold)
            .k_proportional(self.config.k_proportional)
            .k_integral(self.config.k_integral)
            .build();

        let think_result = lp.run(oracle).await;
        let converged = think_result.converged || (has_kb && think_result.final_error >= 0.99);

        let cold_corpus: String = lp
            .cold
            .all()
            .iter()
            .map(|e| e.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let mut chunks_added = 0;
        for entry in lp.cold.all() {
            chunks_added += index.ingest_text("think", &entry.content);
        }

        ThinkLoopResult {
            converged,
            steps: think_result.steps,
            final_error: think_result.final_error,
            cold_corpus,
            chunks_added_to_kb: chunks_added,
        }
    }

    fn retrieve_knowledge(
        &self,
        _prompt: &str,
        topic: &str,
        _prompt_mode: ReasoningMode,
        index: &KnowledgeIndex,
        cold_corpus: &str,
        run_seed: u64,
    ) -> KbRetrievalResult {
        let kb_query = topic.to_string();
        let top_k = self.config.top_k_min + (run_seed % self.config.top_k_var) as usize;
        let kb_chunks: Vec<String> = index
            .query(&kb_query, top_k)
            .into_iter()
            .map(|c| c.text.clone())
            .collect();

        let kb_answer = index
            .answer(&kb_query, top_k)
            .filter(|a| a.split_whitespace().count() >= self.config.min_kb_words)
            .unwrap_or_default();

        let answer_text = if !kb_answer.is_empty() {
            kb_answer.clone()
        } else if !kb_chunks.is_empty() {
            kb_chunks.join(" ")
        } else if cold_corpus.split_whitespace().count() >= 20 {
            cold_corpus.to_string()
        } else {
            let fallback_phrases = [
                "is a foundational topic with many interconnected aspects.",
                "spans many dimensions of knowledge and ongoing research.",
                "continues to be studied and discussed across disciplines.",
                "has shaped thought and practice in significant ways.",
                "is widely explored across academic and professional fields.",
            ];
            let fallback_idx = (run_seed as usize) % fallback_phrases.len();
            format!("{} {}", topic, fallback_phrases[fallback_idx])
        };

        let answer_found = !kb_answer.is_empty() || !kb_chunks.is_empty();

        let answer_quality = is_quality_answer(&answer_text);
        let intent_relevant = true;

        KbRetrievalResult {
            answer_found,
            answer_quality,
            intent_relevant,
            answer_source: if !kb_answer.is_empty() {
                AnswerSource::KnowledgeBase
            } else if !kb_chunks.is_empty() {
                AnswerSource::SymbolicFallback
            } else {
                AnswerSource::ColdCorpusSummary
            },
            topic: topic.to_string(),
            kb_query: topic.to_string(),
            top_k: kb_chunks.len(),
            kb_chunks,
            answer_text,
        }
    }

    fn run_expert_inference(
        &self,
        prompt: &str,
        kb_chunks: &[String],
        depth: usize,
        cf_threshold: f64,
        kb_answer: &str,
        run_seed: u64,
    ) -> Option<ExpertResult> {
        let _ = kb_chunks;
        let chunk_refs: Vec<&str> = kb_chunks.iter().map(|s| s.as_str()).collect();
        let rule_sources: Vec<&str> = chunk_refs
            .iter()
            .copied()
            .take(self.config.rule_source_limit)
            .collect();
        let rule_base = RuleBase::from_chunks(&rule_sources);

        if rule_base.rules.len() < self.config.min_rule_count {
            return None;
        }

        let engine = InferenceEngine::new(depth, cf_threshold);
        let result = engine.reason(prompt, &rule_base)?;

        let builder = ProseBuilder::new(
            self.config.prose_min_sentences,
            self.config.prose_max_sentences,
        );
        let expert_prose = builder.build(&result, &chunk_refs, run_seed);

        if expert_prose.split_whitespace().count() < self.config.min_prose_words {
            return None;
        }

        let final_prose = if !kb_answer.is_empty() {
            let deduplicator = GlobalDeduplicator::new(self.config.dedup_threshold);
            let merged = format!("{}\n\n{}", expert_prose, kb_answer);
            deduplicator.deduplicate_text(&merged)
        } else {
            expert_prose
        };

        Some(ExpertResult {
            conclusion: result.conclusion,
            certainty: result.certainty,
            depth: result.depth,
            rule_count: rule_base.rules.len(),
            trace: result.trace.to_prose(),
            prose: final_prose,
        })
    }

    fn merge_expert_response(
        &self,
        answer_text: String,
        expert_result: &Option<ExpertResult>,
        _kb_chunks: &[String],
        run_seed: u64,
    ) -> String {
        if let Some(result) = expert_result {
            if self.config.use_assembly {
                self.assemble_text(&result.prose, &answer_text, run_seed)
            } else {
                result.prose.clone()
            }
        } else {
            if self.config.use_assembly && !answer_text.is_empty() {
                self.assemble_text(&answer_text, "", run_seed)
            } else {
                answer_text
            }
        }
    }

    fn assemble_text(&self, text: &str, topic: &str, seed: u64) -> String {
        let topic_word: String = topic
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
                            | "that"
                            | "this"
                            | "from"
                            | "into"
                            | "over"
                    )
            })
            .unwrap_or("")
            .to_lowercase();

        let sentences: Vec<String> = text
            .split('\n')
            .flat_map(|line| {
                line.split(". ")
                    .map(|s| s.trim().trim_end_matches('.').trim().to_string())
                    .filter(|s| s.split_whitespace().count() >= 6)
            })
            .map(|s| self.strip_discourse_prefix(&s))
            .filter(|s| SentenceFilter::is_acceptable(s))
            .filter(|s| self.is_on_topic(s, &topic_word))
            .collect();

        if sentences.is_empty() {
            return text.to_string();
        }

        let mut shuffled = sentences;
        let n = shuffled.len();
        let mut rng = seed;
        for i in (1..n).rev() {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (rng >> 33) as usize % (i + 1);
            shuffled.swap(i, j);
        }

        let deduplicator = GlobalDeduplicator::new(self.config.dedup_threshold);
        let mut unique = deduplicator.deduplicate(shuffled);

        if unique.is_empty() {
            return text.to_string();
        }

        if let Some(pos) = unique
            .iter()
            .position(|s| SentenceFilter::is_good_opener(s))
        {
            unique.swap(0, pos);
        }

        ParagraphShaper::new(2, 2, 5, 40, 180).shape(&unique, seed)
    }

    fn strip_discourse_prefix(&self, s: &str) -> String {
        const CONNECTORS: &[&str] = &[
            "Furthermore, ",
            "In addition, ",
            "Notably, ",
            "Moreover, ",
            "As a result, ",
            "In practice, ",
            "Interestingly, ",
            "Consequently, ",
            "Building on this, ",
            "In summary, ",
            "To summarize, ",
            "In conclusion, ",
            "Broadly speaking, ",
            "By design, ",
            "In essence, ",
            "In technical terms, ",
        ];
        for c in CONNECTORS {
            if let Some(rest) = s.strip_prefix(c) {
                let trimmed = rest.trim_start();
                if trimmed.is_empty() {
                    continue;
                }
                let mut out = trimmed.to_string();
                if let Some(first) = out.get_mut(0..1) {
                    first.make_ascii_uppercase();
                }
                return out;
            }
        }
        s.to_string()
    }

    fn is_on_topic(&self, sentence: &str, topic_word: &str) -> bool {
        if topic_word.is_empty() {
            return true;
        }
        let first = match sentence.split_whitespace().next() {
            Some(w) => w,
            None => return true,
        };
        if first.len() < 4 || !first.chars().all(|c| c.is_alphabetic()) {
            return true;
        }
        let mut chars = first.chars();
        let starts_upper = chars.next().map(|c| c.is_uppercase()).unwrap_or(false);
        let rest_lower = chars.all(|c| c.is_lowercase());
        if !starts_upper || !rest_lower {
            return true;
        }
        let first_lower = first.to_lowercase();
        if first_lower.contains(topic_word) || topic_word.contains(&first_lower) {
            return true;
        }
        sentence.to_lowercase().contains(topic_word)
    }

    fn apply_stochastic_enhancement(&self, text: &str) -> StochasticResult {
        let effective_prob = self.config.effective_probability();
        let enhanced = StochasticEnhancer::new(effective_prob)
            .curated_only()
            .enhance_detailed(text);

        StochasticResult {
            text: enhanced.text,
            substitutions: enhanced.substitutions,
            language: format!("{:?}", enhanced.language),
            probability: effective_prob,
        }
    }

    fn build_creative_response(
        &self,
        prompt: &str,
        topic: &str,
        kb_chunks: &[String],
        enhanced_text: &str,
    ) -> String {
        let lang =
            TaskDecomposer::extract_code_language(prompt).unwrap_or_else(|| "text".to_string());
        let kb_corpus: String = kb_chunks.join(" ");
        let task_desc = topic.to_string();
        let code = CodeResponder::generate(&lang, &task_desc, &kb_corpus);
        let code_block = CodeResponder::format_code_block(&lang, &code);
        let frame = ResponseFramer::opening_phrase(
            InputClassifier::classify(prompt),
            topic,
            prompt,
            enhanced_text,
        );

        if frame.is_empty() {
            format!("{}\n\n{}", enhanced_text.trim(), code_block)
        } else {
            format!("{}\n\n{}", frame.trim_end(), code_block)
        }
    }

    fn build_framed_response(
        &self,
        prompt: &str,
        prompt_mode: ReasoningMode,
        topic: &str,
        enhanced_text: &str,
        _run_seed: u64,
    ) -> String {
        let framing_topic = topic.to_string();
        let framed = ResponseFramer::frame(prompt_mode, &framing_topic, prompt, enhanced_text);

        if matches!(prompt_mode, ReasoningMode::Analogical) {
            restructure_analogical_response(&framed)
        } else {
            framed
        }
    }
}
