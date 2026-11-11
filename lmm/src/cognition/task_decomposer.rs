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
                let lang =
                    Self::extract_code_language(prompt).unwrap_or_else(|| "unknown".to_string());
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
                    kind: TaskKind::WebSearch(format!("advantages of {}", topic)),
                    query: format!("advantages of {}", topic),
                    priority: 0,
                },
                SubTask {
                    kind: TaskKind::WebSearch(format!("benefits of {}", topic)),
                    query: format!("benefits of {}", topic),
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
                    kind: TaskKind::WebSearch(format!("how to {}", topic)),
                    query: format!("how to {}", topic),
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
        static CREATIVE_VERBS: &[&str] = &[
            "write ",
            "generate ",
            "create ",
            "implement ",
            "code ",
            "build ",
            "make ",
            "produce ",
            "develop ",
            "program ",
            "script ",
        ];
        let lower = prompt.to_ascii_lowercase();
        CREATIVE_VERBS.iter().any(|v| lower.starts_with(v))
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
        static LANGUAGES: &[(&str, &str)] = &[
            ("python", "python"),
            ("rust", "rust"),
            ("javascript", "javascript"),
            ("typescript", "typescript"),
            ("java", "java"),
            ("kotlin", "kotlin"),
            ("golang", "go"),
            (" go ", "go"),
            ("c++", "cpp"),
            ("cpp", "cpp"),
            ("c#", "csharp"),
            ("csharp", "csharp"),
            ("ruby", "ruby"),
            ("php", "php"),
            ("swift", "swift"),
            ("scala", "scala"),
            ("haskell", "haskell"),
            ("elixir", "elixir"),
            ("erlang", "erlang"),
            ("clojure", "clojure"),
            ("lua", "lua"),
            ("perl", "perl"),
            ("bash", "bash"),
            ("shell", "bash"),
            ("julia", "julia"),
            ("dart", "dart"),
            ("zig", "zig"),
        ];
        let lower = prompt.to_ascii_lowercase();
        for &(pattern, canonical) in LANGUAGES {
            if lower.contains(pattern) {
                return Some(canonical.to_string());
            }
        }
        None
    }
}

/// Extracts a simple topic noun phrase from a prompt by stripping leading verbs/question words.
///
/// # Time Complexity
///
/// O(|prompt|)
fn extract_simple_topic(prompt: &str) -> String {
    static STRIP_PREFIXES: &[&str] = &[
        "what are the advantages of ",
        "what are the disadvantages of ",
        "what are the pros and cons of ",
        "what are the ",
        "what are ",
        "what is the ",
        "what is ",
        "how to ",
        "why does ",
        "why is ",
        "why are ",
        "explain ",
        "describe ",
        "define ",
        "write ",
        "generate ",
        "create ",
        "implement ",
        "make ",
        "build ",
        "show me ",
        "give me ",
    ];
    let lower = prompt.to_ascii_lowercase();
    for prefix in STRIP_PREFIXES {
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
    static SPLITTERS: &[&str] = &[" vs ", " versus ", " compared to "];
    let lower = prompt.to_ascii_lowercase();
    for &splitter in SPLITTERS {
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
    static CREATIVE_PREFIXES: &[&str] = &[
        "write ",
        "generate ",
        "create ",
        "implement ",
        "make ",
        "build ",
        "show me ",
        "give me ",
        "produce ",
        "develop ",
        "program ",
    ];
    let lower = prompt.to_ascii_lowercase();
    let stripped = CREATIVE_PREFIXES
        .iter()
        .find_map(|p| lower.strip_prefix(*p))
        .map(|s| &prompt[lower.len() - s.len()..])
        .unwrap_or(prompt);

    static IN_PATTERNS: &[&str] = &[
        " in python",
        " in rust",
        " in javascript",
        " in typescript",
        " in java",
        " in kotlin",
        " in go",
        " in golang",
        " in c++",
        " in cpp",
        " in c#",
        " in csharp",
        " in ruby",
        " in php",
        " in swift",
        " in scala",
        " in haskell",
        " in elixir",
        " in erlang",
        " in clojure",
        " in lua",
        " in perl",
        " in bash",
        " in shell",
        " in julia",
        " in dart",
        " in zig",
    ];
    let lower2 = stripped.to_ascii_lowercase();
    for &pat in IN_PATTERNS {
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
