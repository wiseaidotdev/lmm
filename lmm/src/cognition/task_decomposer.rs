// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Task Decomposer
//!
//! Decomposes a user prompt into a list of concrete [`SubTask`]s, each carrying
//! a typed [`TaskKind`] and a targeted search query.  The decomposition is
//! **mode-aware**: evaluative prompts generate advantages/disadvantages queries,
//! task-execution prompts generate code-example queries, and so on.
//!
//! An intelligent variant, [`TaskDecomposer::decompose_intelligent`], inspects
//! the last reinforcement-learning reward signal and prepends an
//! [`TaskKind::AcquireSkill`] sub-task when the agent needs to build knowledge
//! before answering.
//!
//! ## Examples
//!
//! ```rust
//! use lmm::cognition::task_decomposer::{TaskDecomposer, TaskKind};
//! use lmm::cognition::thought::classifier::ReasoningMode;
//!
//! let tasks = TaskDecomposer::decompose("write hello world in python", ReasoningMode::TaskExecution, 4);
//! assert!(!tasks.is_empty());
//! ```

use crate::cognition::thought::classifier::ReasoningMode;
use crate::constants::task_decomposer::{
    CREATIVE_ESSAY_KEYWORDS, CREATIVE_POEM_KEYWORDS, CREATIVE_STORY_KEYWORDS,
    TASK_ACQUIRE_SKILL_LABEL, TASK_CREATIVE_PREFIXES, TASK_CREATIVE_VERBS, TASK_FALLBACK_LANGUAGE,
    TASK_FMT_ADVANTAGES, TASK_FMT_BENEFITS, TASK_FMT_DEFINITION, TASK_FMT_HOW_TO, TASK_FMT_WHAT_IS,
    TASK_FORM_STRIP_SUFFIXES, TASK_IN_LANG_PATTERNS, TASK_LANGUAGE_STRIP_PREFIXES, TASK_LANGUAGES,
    TASK_LOW_REWARD_THRESHOLD, TASK_MEDIUM_REWARD_THRESHOLD, TASK_MODIFIER_STRIP_PREFIXES,
    TASK_SKILL_REWARD_BOOST, TASK_STRIP_PREFIXES, TASK_VS_SPLITTERS,
};

/// The kind of work a [`SubTask`] represents.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::task_decomposer::TaskKind;
///
/// let _k = TaskKind::WebSearch("rust advantages".to_string());
/// ```
#[derive(Debug, Clone)]
pub enum TaskKind {
    /// A web search with the given query string.
    WebSearch(String),
    /// Generate code in `language` to accomplish `task`.
    CodeGeneration {
        /// Target programming language (e.g. `"python"`, `"rust"`).
        language: String,
        /// Description of what to generate (e.g. `"hello world"`).
        task: String,
    },
    /// Compare two named entities.
    Comparison(String, String),
    /// Look up the definition of a term.
    Definition(String),
    /// Acquire knowledge about `topic` before attempting a full answer.
    ///
    /// Generated when the last RL reward is below
    /// [`TASK_LOW_REWARD_THRESHOLD`][crate::constants::task_decomposer::TASK_LOW_REWARD_THRESHOLD],
    /// signalling that the agent lacks sufficient grounding.
    AcquireSkill {
        /// The topic to research.
        topic: String,
        /// Pre-built search queries ordered by expected relevance.
        search_queries: Vec<String>,
    },
}

/// A single decomposed sub-task with a typed kind and a search query.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::task_decomposer::{SubTask, TaskKind};
///
/// let _t = SubTask {
///     kind: TaskKind::WebSearch("rust".to_string()),
///     query: "rust".to_string(),
///     priority: 0,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SubTask {
    /// What kind of work this task is.
    pub kind: TaskKind,
    /// The search query to use when fetching information for this task.
    pub query: String,
    /// Lower values are higher priority (0 = highest).
    pub priority: u8,
}

/// Describes the kind of creative output a task-execution prompt requests.
///
/// Used by [`TaskDecomposer::detect_creative_kind`] to route creative prompts
/// to the appropriate responder in the assembly stage.
#[derive(Debug, Clone, PartialEq)]
pub enum CreativeOutputKind {
    /// Source code in the given programming language.
    Code {
        /// Canonical language name (e.g. `"python"`, `"rust"`).
        language: String,
    },
    /// A poem, verse, haiku, or similar poetic form.
    Poem,
    /// A story, narrative, or short fiction.
    Story,
    /// An essay, article, or analytical write-up.
    Essay,
    /// A generic creative/prose response.
    Generic,
}

/// Stateless decomposer that maps a prompt and mode into a list of [`SubTask`]s.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::task_decomposer::TaskDecomposer;
/// use lmm::cognition::thought::classifier::ReasoningMode;
///
/// let tasks = TaskDecomposer::decompose("rust vs python", ReasoningMode::Analogical, 4);
/// assert!(tasks.iter().any(|t| t.query.contains("rust")));
/// ```
pub struct TaskDecomposer;

impl TaskDecomposer {
    /// Decomposes `prompt` into at most `max` sub-tasks for the given `mode`.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Raw user prompt.
    /// * `mode`   - Classified reasoning mode.
    /// * `max`    - Maximum number of sub-tasks to return.
    ///
    /// # Time Complexity
    ///
    /// O(|prompt|)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::task_decomposer::TaskDecomposer;
    /// use lmm::cognition::thought::classifier::ReasoningMode;
    ///
    /// let tasks = TaskDecomposer::decompose("what are the advantages of rust?", ReasoningMode::Evaluative, 4);
    /// assert!(!tasks.is_empty());
    /// ```
    pub fn decompose(prompt: &str, mode: ReasoningMode, max: usize) -> Vec<SubTask> {
        let topic = extract_simple_topic(prompt);
        let mut tasks: Vec<SubTask> = match mode {
            ReasoningMode::TaskExecution => {
                let lang = Self::extract_code_language(prompt)
                    .unwrap_or_else(|| TASK_FALLBACK_LANGUAGE.to_string());
                let task_desc = extract_task_description(prompt);
                vec![
                    SubTask {
                        kind: TaskKind::CodeGeneration {
                            language: lang.clone(),
                            task: task_desc.clone(),
                        },
                        query: format!("{} {} example", lang, task_desc),
                        priority: 0,
                    },
                    SubTask {
                        kind: TaskKind::WebSearch(format!(
                            "how to write {} in {}",
                            task_desc, lang
                        )),
                        query: format!("how to write {} in {}", task_desc, lang),
                        priority: 1,
                    },
                    SubTask {
                        kind: TaskKind::WebSearch(format!("{} {} syntax", lang, task_desc)),
                        query: format!("{} {} syntax", lang, task_desc),
                        priority: 2,
                    },
                    SubTask {
                        kind: TaskKind::WebSearch(format!("{} getting started", lang)),
                        query: format!("{} getting started", lang),
                        priority: 3,
                    },
                ]
            }
            ReasoningMode::Evaluative => vec![
                SubTask {
                    kind: TaskKind::WebSearch(format!("{} {}", TASK_FMT_ADVANTAGES, topic)),
                    query: format!("{} {}", TASK_FMT_ADVANTAGES, topic),
                    priority: 0,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("{} {}", TASK_FMT_BENEFITS, topic)),
                    query: format!("{} {}", TASK_FMT_BENEFITS, topic),
                    priority: 1,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("disadvantages of {}", topic)),
                    query: format!("disadvantages of {}", topic),
                    priority: 2,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("pros and cons of {}", topic)),
                    query: format!("pros and cons of {}", topic),
                    priority: 3,
                },
            ],
            ReasoningMode::Causal => vec![
                SubTask {
                    kind: TaskKind::WebSearch(format!("why {}", topic)),
                    query: format!("why {}", topic),
                    priority: 0,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("{} mechanism", topic)),
                    query: format!("{} mechanism", topic),
                    priority: 1,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("{} causes", topic)),
                    query: format!("{} causes", topic),
                    priority: 2,
                },
            ],
            ReasoningMode::Analogical => {
                let entities = extract_vs_entities(prompt);
                if let Some((a, b)) = entities {
                    vec![
                        SubTask {
                            kind: TaskKind::Comparison(a.clone(), b.clone()),
                            query: format!("{} vs {} comparison", a, b),
                            priority: 0,
                        },
                        SubTask {
                            kind: TaskKind::WebSearch(format!("{} {} differences", a, b)),
                            query: format!("{} {} differences", a, b),
                            priority: 1,
                        },
                        SubTask {
                            kind: TaskKind::WebSearch(format!("{} {} similarities", a, b)),
                            query: format!("{} {} similarities", a, b),
                            priority: 2,
                        },
                    ]
                } else {
                    vec![SubTask {
                        kind: TaskKind::WebSearch(topic.clone()),
                        query: topic,
                        priority: 0,
                    }]
                }
            }
            ReasoningMode::Procedural => vec![
                SubTask {
                    kind: TaskKind::WebSearch(format!("{} {}", TASK_FMT_HOW_TO, topic)),
                    query: format!("{} {}", TASK_FMT_HOW_TO, topic),
                    priority: 0,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("{} tutorial", topic)),
                    query: format!("{} tutorial", topic),
                    priority: 1,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("{} guide", topic)),
                    query: format!("{} guide", topic),
                    priority: 2,
                },
            ],
            ReasoningMode::Definitional => vec![
                SubTask {
                    kind: TaskKind::Definition(topic.clone()),
                    query: format!("what is {}", topic),
                    priority: 0,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("{} definition", topic)),
                    query: format!("{} definition", topic),
                    priority: 1,
                },
            ],
            ReasoningMode::FactualLookup | ReasoningMode::Exploratory => vec![SubTask {
                kind: TaskKind::WebSearch(topic.clone()),
                query: topic,
                priority: 0,
            }],
        };
        tasks.truncate(max);
        tasks
    }

    /// Decomposes `prompt` intelligently, prepending an [`TaskKind::AcquireSkill`]
    /// task when `last_reward` is below the low-reward threshold, and injecting
    /// definitional refinement queries when it falls in the medium range.
    ///
    /// This is the preferred entry point for the TRAM pipeline because it adapts
    /// to the RL reward signal: when the agent has low confidence in its knowledge
    /// of the topic, it schedules a knowledge-acquisition pass before answering;
    /// when confidence is moderate it supplements with contextual definition queries.
    ///
    /// # Arguments
    ///
    /// * `prompt`      - Raw user prompt.
    /// * `mode`        - Classified reasoning mode.
    /// * `max`         - Maximum number of sub-tasks to return.
    /// * `last_reward` - Most-recent episode reward from the learning engine
    ///   (range 0.0-1.0).  Values below
    ///   [`TASK_LOW_REWARD_THRESHOLD`] trigger skill acquisition; values between
    ///   that and [`TASK_MEDIUM_REWARD_THRESHOLD`] add definitional refinement.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::task_decomposer::TaskDecomposer;
    /// use lmm::cognition::thought::classifier::ReasoningMode;
    ///
    /// let tasks = TaskDecomposer::decompose_intelligent(
    ///     "what are the advantages of rust?",
    ///     ReasoningMode::Evaluative,
    ///     6,
    ///     0.1,
    /// );
    /// // Low reward → first task is AcquireSkill.
    /// assert!(matches!(
    ///     tasks[0].kind,
    ///     lmm::cognition::task_decomposer::TaskKind::AcquireSkill { .. }
    /// ));
    /// ```
    pub fn decompose_intelligent(
        prompt: &str,
        mode: ReasoningMode,
        max: usize,
        last_reward: f64,
    ) -> Vec<SubTask> {
        let mut base = Self::decompose(prompt, mode, max.saturating_sub(1));

        if last_reward < TASK_LOW_REWARD_THRESHOLD {
            let topic = if matches!(mode, ReasoningMode::TaskExecution) {
                let raw = extract_simple_topic(prompt);
                strip_code_subject_noise(&raw)
            } else {
                extract_simple_topic(prompt)
            };
            let acquire = SubTask {
                kind: TaskKind::AcquireSkill {
                    topic: topic.clone(),
                    search_queries: vec![
                        format!("{} {}", TASK_FMT_ADVANTAGES, topic),
                        format!("{} overview", topic),
                        format!("{} introduction", topic),
                    ],
                },
                query: format!(
                    "{} {} {}",
                    TASK_ACQUIRE_SKILL_LABEL, topic, TASK_SKILL_REWARD_BOOST
                ),
                priority: 0,
            };
            for t in base.iter_mut() {
                t.priority = t.priority.saturating_add(1);
            }
            base.insert(0, acquire);
        }

        if (TASK_LOW_REWARD_THRESHOLD..TASK_MEDIUM_REWARD_THRESHOLD).contains(&last_reward) {
            let topic = if matches!(mode, ReasoningMode::TaskExecution) {
                let raw = extract_simple_topic(prompt);
                strip_code_subject_noise(&raw)
            } else {
                extract_simple_topic(prompt)
            };
            let refinement_queries = vec![
                format!("{} {}", TASK_FMT_WHAT_IS, topic),
                format!("{} {}", topic, TASK_FMT_DEFINITION),
                format!("{} examples", topic),
            ];
            for q in refinement_queries {
                if base.len() < max {
                    base.push(SubTask {
                        kind: TaskKind::WebSearch(q.clone()),
                        query: q,
                        priority: 2,
                    });
                }
            }
        }

        base.truncate(max);
        base
    }

    /// Returns `true` when the prompt describes a creative/generative task.
    ///
    /// Detects verbs like "write", "generate", "create", "implement",
    /// "code", "build" appearing at or near the start of the prompt.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::task_decomposer::TaskDecomposer;
    ///
    /// assert!(TaskDecomposer::is_creative_task("write hello world in python"));
    /// assert!(TaskDecomposer::is_creative_task("generate a fibonacci function"));
    /// assert!(!TaskDecomposer::is_creative_task("what is rust?"));
    /// ```
    pub fn is_creative_task(prompt: &str) -> bool {
        let lower = prompt.to_ascii_lowercase();
        TASK_CREATIVE_VERBS.iter().any(|v| lower.starts_with(v))
    }

    /// Extracts the target programming language from a prompt like "... in python".
    ///
    /// Returns `None` when no language keyword is found.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::task_decomposer::TaskDecomposer;
    ///
    /// assert_eq!(
    ///     TaskDecomposer::extract_code_language("write hello world in python"),
    ///     Some("python".to_string())
    /// );
    /// assert_eq!(
    ///     TaskDecomposer::extract_code_language("rust program"),
    ///     Some("rust".to_string())
    /// );
    /// assert_eq!(TaskDecomposer::extract_code_language("hello world"), None);
    /// ```
    pub fn extract_code_language(prompt: &str) -> Option<String> {
        let lower = prompt.to_ascii_lowercase();
        for &(pattern, canonical) in TASK_LANGUAGES {
            if lower.contains(pattern) {
                return Some(canonical.to_string());
            }
        }
        None
    }

    /// Detects the kind of creative output a task-execution prompt is requesting.
    ///
    /// Checks in order: programming language keywords → [`CreativeOutputKind::Code`];
    /// poem keywords → [`CreativeOutputKind::Poem`];
    /// story keywords → [`CreativeOutputKind::Story`];
    /// essay keywords → [`CreativeOutputKind::Essay`];
    /// otherwise → [`CreativeOutputKind::Generic`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::task_decomposer::{CreativeOutputKind, TaskDecomposer};
    ///
    /// assert!(matches!(
    ///     TaskDecomposer::detect_creative_kind("write a python factorial program"),
    ///     CreativeOutputKind::Code { .. }
    /// ));
    /// assert_eq!(
    ///     TaskDecomposer::detect_creative_kind("write a poem about life"),
    ///     CreativeOutputKind::Poem
    /// );
    /// ```
    pub fn detect_creative_kind(prompt: &str) -> CreativeOutputKind {
        let lower = prompt.to_ascii_lowercase();
        if CREATIVE_POEM_KEYWORDS.iter().any(|&kw| lower.contains(kw)) {
            return CreativeOutputKind::Poem;
        }
        if CREATIVE_STORY_KEYWORDS.iter().any(|&kw| lower.contains(kw)) {
            return CreativeOutputKind::Story;
        }
        if CREATIVE_ESSAY_KEYWORDS.iter().any(|&kw| lower.contains(kw)) {
            return CreativeOutputKind::Essay;
        }
        if Self::is_creative_task(prompt)
            && let Some(lang) = Self::extract_code_language(prompt)
        {
            return CreativeOutputKind::Code { language: lang };
        }
        CreativeOutputKind::Generic
    }
}

/// Strips language, modifier, and form-noun noise from a code task subject phrase.
///
/// Uses [`TASK_LANGUAGE_STRIP_PREFIXES`], [`TASK_MODIFIER_STRIP_PREFIXES`], and
/// [`TASK_FORM_STRIP_SUFFIXES`] from the constants crate to avoid a circular
/// dependency with `tram::query::extract_creative_subject`.
///
/// # Examples
///
/// * `"simple python hello world program"` → `"hello world"`
/// * `"python factorial function"` → `"factorial"`
/// * `"derivative of a number"` → `"derivative of a number"` (unchanged)
///
/// # Time Complexity
///
/// O(|s| × (|prefixes| + |suffixes|))
fn strip_code_subject_noise(s: &str) -> String {
    let mut result = s.to_string();
    let mut changed = true;
    while changed {
        changed = false;
        for &prefix in TASK_LANGUAGE_STRIP_PREFIXES {
            if result.starts_with(prefix) {
                result = result[prefix.len()..].to_string();
                changed = true;
                break;
            }
        }
        for &prefix in TASK_MODIFIER_STRIP_PREFIXES {
            if result.starts_with(prefix) {
                result = result[prefix.len()..].to_string();
                changed = true;
                break;
            }
        }
    }
    for &suffix in TASK_FORM_STRIP_SUFFIXES {
        if result.ends_with(suffix) {
            result = result[..result.len() - suffix.len()].to_string();
            break;
        }
    }
    result.trim().to_string()
}

/// Extracts a simple topic noun phrase from a prompt by stripping leading verbs/question words.
///
/// # Time Complexity
///
/// O(|prompt|)
fn extract_simple_topic(prompt: &str) -> String {
    let lower = prompt.to_ascii_lowercase();
    for prefix in TASK_STRIP_PREFIXES {
        if lower.starts_with(prefix) {
            return prompt[prefix.len()..]
                .trim()
                .trim_end_matches('?')
                .trim_end_matches('.')
                .trim()
                .to_string();
        }
    }
    prompt
        .trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim()
        .to_string()
}

/// Extracts two entity names from a comparison prompt ("rust vs python" → ("rust", "python")).
///
/// # Time Complexity
///
/// O(|prompt|)
fn extract_vs_entities(prompt: &str) -> Option<(String, String)> {
    let lower = prompt.to_ascii_lowercase();
    for &splitter in TASK_VS_SPLITTERS {
        if let Some(pos) = lower.find(splitter) {
            let a = extract_simple_topic(&prompt[..pos]);
            let b = prompt[pos + splitter.len()..]
                .trim()
                .trim_end_matches('?')
                .trim()
                .to_string();
            if !a.is_empty() && !b.is_empty() {
                return Some((a, b));
            }
        }
    }
    None
}

/// Extracts the task description from a creative prompt.
///
/// For "write hello world in python" returns "hello world".
///
/// # Time Complexity
///
/// O(|prompt|)
fn extract_task_description(prompt: &str) -> String {
    let lower = prompt.to_ascii_lowercase();
    let stripped = TASK_CREATIVE_PREFIXES
        .iter()
        .find_map(|p| lower.strip_prefix(*p))
        .map(|s| &prompt[lower.len() - s.len()..])
        .unwrap_or(prompt);

    let lower2 = stripped.to_ascii_lowercase();
    for &pat in TASK_IN_LANG_PATTERNS {
        if let Some(pos) = lower2.find(pat) {
            return stripped[..pos].trim().to_string();
        }
    }
    stripped
        .trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim()
        .to_string()
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
