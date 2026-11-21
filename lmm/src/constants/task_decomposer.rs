// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Task Decomposer Constants

/// Verbs at the beginning of a prompt that signal a creative/generative task.
pub static TASK_CREATIVE_VERBS: &[&str] = &[
    "show me an ",
    "show me a ",
    "show me ",
    "give me an ",
    "give me a ",
    "give me ",
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

/// Mapping of raw language keywords (lower-cased) to their canonical names.
///
/// Each tuple is `(keyword_in_prompt, canonical_language_name)`.
pub static TASK_LANGUAGES: &[(&str, &str)] = &[
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

/// Prefixes stripped from a prompt before extracting its simple topic noun phrase.
pub static TASK_STRIP_PREFIXES: &[&str] = &[
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

/// Separators used to split a comparison prompt into two entity names.
pub static TASK_VS_SPLITTERS: &[&str] = &[" vs ", " versus ", " compared to "];

/// Keywords inside a prompt that identify the output as a poem or verse.
pub const CREATIVE_POEM_KEYWORDS: &[&str] = &[
    "poem", "verse", "sonnet", "haiku", "stanza", "rhyme", "lyric", "poetry", "ode", "ballad",
];

/// Keywords inside a prompt that identify the output as a story or narrative.
pub const CREATIVE_STORY_KEYWORDS: &[&str] = &[
    "story",
    "narrative",
    "tale",
    "fiction",
    "short story",
    "novella",
];

/// Keywords inside a prompt that identify the output as an essay or article.
pub const CREATIVE_ESSAY_KEYWORDS: &[&str] = &[
    "essay",
    "article",
    "report",
    "writeup",
    "write-up",
    "analysis",
    "blog post",
];

/// Verb prefixes that introduce the task body in a creative prompt.
pub static TASK_CREATIVE_PREFIXES: &[&str] = &[
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

/// Patterns that mark the language suffix in a creative prompt (e.g. `" in python"`).
pub static TASK_IN_LANG_PATTERNS: &[&str] = &[
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

/// Label used for the `AcquireSkill` sub-task kind in log output and serialisation.
pub const TASK_ACQUIRE_SKILL_LABEL: &str = "learn-skill";

/// Reward boost applied to `AcquireSkill` sub-tasks to give them higher priority
/// when the learning engine has not yet mastered the topic.
pub const TASK_SKILL_REWARD_BOOST: f64 = 0.3;

/// Minimum character length a topic word must have to be considered non-trivial.
pub const TASK_MIN_TOPIC_WORD_LEN: usize = 3;

/// Default upper bound on the number of sub-tasks generated per decomposition.
pub const TASK_MAX_SUB_TASKS: usize = 6;

/// Query prefix used for the evaluative advantages sub-task.
pub const TASK_FMT_ADVANTAGES: &str = "advantages of";

/// Query prefix used for the evaluative benefits sub-task.
pub const TASK_FMT_BENEFITS: &str = "benefits of";

/// Query prefix used for the procedural how-to sub-task.
pub const TASK_FMT_HOW_TO: &str = "how to";

/// Fallback language name when no language keyword is detected in a prompt.
pub const TASK_FALLBACK_LANGUAGE: &str = "unknown";

/// Reward threshold below which an `AcquireSkill` sub-task is prepended.
pub const TASK_LOW_REWARD_THRESHOLD: f64 = 0.3;

/// Reward threshold below which additional refinement queries are added.
///
/// When the last episode reward falls between [`TASK_LOW_REWARD_THRESHOLD`] and
/// this value the decomposer supplements the query list with definitional and
/// contextual sub-tasks to broaden KB coverage before the main response is built.
pub const TASK_MEDIUM_REWARD_THRESHOLD: f64 = 0.6;

/// Query format string for the \"what is X\" definitional refinement sub-task.
pub const TASK_FMT_WHAT_IS: &str = "what is";

/// Query format string for the \"X definition\" definitional refinement sub-task.
pub const TASK_FMT_DEFINITION: &str = "definition";

/// Programming language name prefixes for stripping noise from code task subjects.
///
/// Intentionally mirrors
/// [`SUBJECT_LANGUAGE_PREFIXES`][crate::constants::tram_query::SUBJECT_LANGUAGE_PREFIXES]
/// to avoid a circular dependency between `cognition/task_decomposer` and `tram/query`.
pub const TASK_LANGUAGE_STRIP_PREFIXES: &[&str] = &[
    "python ",
    "rust ",
    "javascript ",
    "typescript ",
    "java ",
    "kotlin ",
    "go ",
    "golang ",
    "c++ ",
    "cpp ",
    "c# ",
    "csharp ",
    "ruby ",
    "php ",
    "swift ",
    "scala ",
    "haskell ",
    "elixir ",
    "erlang ",
    "clojure ",
    "lua ",
    "perl ",
    "bash ",
    "shell ",
    "julia ",
    "dart ",
    "zig ",
];

/// Generic adjective modifier prefixes for stripping noise from code task subjects.
///
/// Intentionally mirrors
/// [`SUBJECT_MODIFIER_PREFIXES`][crate::constants::tram_query::SUBJECT_MODIFIER_PREFIXES]
/// to avoid a circular dependency between `cognition/task_decomposer` and `tram/query`.
/// Articles (`"a "`, `"an "`, `"the "`) are included here so that raw topic strings
/// produced by `extract_simple_topic` (e.g. `"a simple python hello world program"`) are
/// fully cleaned before the subject is used for Wikipedia lookups or KB retrieval.
pub const TASK_MODIFIER_STRIP_PREFIXES: &[&str] = &[
    "a ",
    "an ",
    "the ",
    "simple ",
    "basic ",
    "easy ",
    "small ",
    "quick ",
    "minimal ",
    "tiny ",
    "straightforward ",
    "beginner ",
    "introductory ",
    "classic ",
];

/// Output-type suffixes for stripping noise from code task subjects.
///
/// Intentionally mirrors
/// [`SUBJECT_FORM_SUFFIXES`][crate::constants::tram_query::SUBJECT_FORM_SUFFIXES]
/// to avoid a circular dependency between `cognition/task_decomposer` and `tram/query`.
pub const TASK_FORM_STRIP_SUFFIXES: &[&str] = &[
    " program",
    " script",
    " function",
    " code",
    " app",
    " application",
    " module",
    " class",
    " snippet",
    " example",
    " implementation",
    " algorithm",
];

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
