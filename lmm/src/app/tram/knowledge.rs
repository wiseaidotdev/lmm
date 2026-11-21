// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Knowledge Stage
//!
//! Loads the KB from disk, runs the web pre-fetch, consults the skill registry,
//! and decomposes the prompt into sub-tasks.  The populated [`KnowledgeIndex`]
//! and the raw [`WebFetchResult`] are forwarded to later pipeline stages.

#[cfg(feature = "cli")]
use {
    crate::{
        cli::tram::{
            prefetch_web_for_query,
            query::{extract_creative_subject, extract_topic},
        },
        cognition::{
            ReasoningMode,
            knowledge::{KnowledgeIndex, KnowledgeSource, ingest as cognition_ingest},
            learning::{LearningConfig, engine::LearningEngine},
            skill::{Skill, SkillRegistry},
            task_decomposer::{SubTask, TaskDecomposer, TaskKind},
            thought::classifier::InputClassifier,
        },
        constants::skill::{SKILL_SOURCE_LABEL, SKILL_VERSION_INITIAL},
        constants::tram_pipeline::{SKILL_SYNTHESIS_TOP_K, SKILL_SYNTHESISED_CONFIDENCE},
        tram::{config::WebFetchResult, web_fetch::prefetch_web},
    },
    std::{
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    },
    tracing::{error, info},
};

/// Data produced by the knowledge stage and consumed by subsequent stages.
#[cfg(feature = "cli")]
pub struct KnowledgeStageOutput {
    /// Populated knowledge index ready for retrieval.
    pub index: KnowledgeIndex,
    /// Raw web-fetch result, used in the assembly stage for the URL footer.
    pub web_result: WebFetchResult,
    /// Whether the caller supplied an explicit KB path.
    pub kb_provided: bool,
    /// Sub-tasks decomposed from the prompt.
    pub sub_tasks: Vec<SubTask>,
    /// Skill registry (mutable so the assembly stage can flush it after the episode).
    pub skill_registry: SkillRegistry,
    /// Learning engine (mutable so the assembly stage can record the episode reward).
    pub learning_engine: LearningEngine,
    /// Prompt reasoning mode classified by [`InputClassifier`].
    pub prompt_mode: ReasoningMode,
    /// Directory used for skill and learning-engine state persistence.
    pub skill_dir: PathBuf,
}

/// Runs the knowledge stage.
///
/// 1. Loads the optional local KB file or directory into `index`.
/// 2. Pre-fetches web results (DuckDuckGo) and ingests them into `index`.
/// 3. Looks up the skill registry for a cached synthesis of the topic.
/// 4. Decomposes the prompt into sub-tasks and acquires skills for each one.
///
/// # Arguments
///
/// * `prompt`         - The raw user prompt.
/// * `knowledge_base` - Optional path to a local KB file or directory.
/// * `deep`           - Whether deep-scrape web enrichment is active.
/// * `verbose`        - Emit diagnostic log messages when `true`.
/// * `max_tasks`      - Upper bound on task-decomposer sub-tasks.
/// * `skill_dir`      - Filesystem directory for persistent skill storage.
///
/// # Returns
///
/// A fully populated [`KnowledgeStageOutput`].
///
/// # Time Complexity
///
/// O(|KB| + |web_results|)
#[cfg(feature = "cli")]
pub async fn run_knowledge_stage(
    prompt: &str,
    knowledge_base: Option<String>,
    deep: bool,
    verbose: bool,
    max_tasks: usize,
    skill_dir: PathBuf,
) -> KnowledgeStageOutput {
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

    let web_result = prefetch_web(prompt, &mut index, kb_provided, deep, verbose).await;

    let mut skill_registry = SkillRegistry::load_or_create(&skill_dir);
    let learning_engine = LearningEngine::load_or_new(LearningConfig::default(), &skill_dir);

    let prompt_mode = InputClassifier::classify(prompt);
    let last_reward = if learning_engine.episode_count() > 0 {
        learning_engine.total_reward() / learning_engine.episode_count() as f64
    } else {
        1.0
    };
    let sub_tasks =
        TaskDecomposer::decompose_intelligent(prompt, prompt_mode, max_tasks, last_reward);
    let topic_for_skill = if matches!(prompt_mode, ReasoningMode::TaskExecution) {
        extract_creative_subject(prompt)
    } else {
        extract_topic(prompt)
    };

    if let Some(skill) = skill_registry.get(&topic_for_skill) {
        if verbose {
            info!(
                "  🧠 Skill cache hit: topic='{}' confidence={:.2}",
                topic_for_skill, skill.confidence
            );
        }
        index.ingest_text(SKILL_SOURCE_LABEL, &skill.content);
    }

    for task in &sub_tasks {
        match &task.kind {
            TaskKind::WebSearch(q) => {
                prefetch_web_for_query(q, &mut index, verbose).await;
            }
            TaskKind::AcquireSkill {
                topic,
                search_queries,
            } => {
                for sq in search_queries {
                    prefetch_web_for_query(sq, &mut index, verbose).await;
                }
                let kb_text: String = index
                    .query(topic, SKILL_SYNTHESIS_TOP_K)
                    .into_iter()
                    .map(|c| c.text.clone())
                    .collect::<Vec<_>>()
                    .join(" ");
                if !kb_text.is_empty() {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let skill = Skill {
                        topic: topic.clone(),
                        content: kb_text,
                        confidence: SKILL_SYNTHESISED_CONFIDENCE,
                        created_at: now,
                        version: SKILL_VERSION_INITIAL,
                    };
                    skill_registry.upsert(skill);
                    if verbose {
                        info!("  🧠 Skill synthesised: topic='{}'", topic);
                    }
                }
            }
            _ => {}
        }
    }

    KnowledgeStageOutput {
        index,
        web_result,
        kb_provided,
        sub_tasks,
        skill_registry,
        learning_engine,
        prompt_mode,
        skill_dir,
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
