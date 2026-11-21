// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # TRAM Query Constants

use phf::phf_set;

pub const FILLER_PREFIXES: &[&str] = &[
    "what is a ",
    "what is an ",
    "what is the ",
    "what is ",
    "what are the ",
    "what are ",
    "what was ",
    "what were ",
    "how does a ",
    "how does an ",
    "how does ",
    "how do ",
    "how is ",
    "how are ",
    "how to ",
    "explain ",
    "describe ",
    "define ",
    "tell me about ",
    "what's ",
    "whats ",
    "who is ",
    "who was ",
    "why is a ",
    "why is an ",
    "why is the ",
    "why is ",
    "why are ",
    "why does ",
    "why do ",
    "where is ",
    "when is ",
    "give me information about ",
    "give me info on ",
    "write a ",
    "write an ",
    "write ",
    "generate a ",
    "generate an ",
    "create a ",
    "create an ",
    "implement a ",
    "implement an ",
    "code a ",
    "build a ",
    "show me how to ",
    "show me a ",
    "show me ",
    "give me a ",
    "give me an ",
    "compare ",
];

pub const MODE_KEYWORD_PREFIXES: &[&str] = &[
    "advantages of ",
    "advantage of ",
    "benefits of ",
    "benefit of ",
    "disadvantages of ",
    "disadvantage of ",
    "pros and cons of ",
    "pros of ",
    "cons of ",
    "drawbacks of ",
    "drawback of ",
    "strengths of ",
    "strength of ",
    "weaknesses of ",
    "weakness of ",
    "limitations of ",
    "limitation of ",
    "inconveniences of ",
    "inconvenience of ",
    "upsides of ",
    "downsides of ",
    "merits of ",
    "merit of ",
    "demerits of ",
    "demerit of ",
    "causes of ",
    "cause of ",
    "reason for ",
    "reasons for ",
    "explanation for ",
    "effect of ",
    "effects of ",
    "definition of ",
    "meaning of ",
    "concept of ",
    "examples of ",
    "overview of ",
    "history of ",
    "installation of ",
    "setup of ",
    "usage of ",
    "use of ",
    "uses of ",
    "comparison of ",
    "difference between ",
    "differences between ",
    "similarities between ",
    "relationship between ",
    "why is ",
    "why are ",
    "why does ",
    "why do ",
];

pub const PREDICATE_WORDS: &[&str] = &[
    "not", "no", "never", "have", "has", "had", "is", "are", "was", "were", "do", "does", "did",
    "will", "would", "could", "should", "can", "cannot", "work", "works", "make", "makes", "use",
    "uses", "need", "needs", "allow", "allows", "prevent", "prevents", "require", "requires",
    "cause", "causes",
];

pub const TRAILING_VERBS: &[&str] = &[
    " work",
    " works",
    " function",
    " functions",
    " operate",
    " operates",
    " run",
    " runs",
    "do",
    "does",
    " mean",
    " means",
    " help",
    "helps",
    "differ",
    "differs",
];

pub const CONNECTOR_PREFIXES: &[&str] = &[
    "Furthermore, ",
    "In addition, ",
    "Notably, ",
    "Moreover, ",
    "As a result, ",
    "In practice, ",
    "Interestingly, ",
    "Consequently, ",
    "Building on this, ",
    "To elaborate, ",
    "More specifically, ",
    "On the technical side, ",
    "From a practical standpoint, ",
    "In this context, ",
    "Also, ",
    "However, ",
    "Therefore, ",
    "Thus, ",
    "Hence, ",
    "More precisely, ",
    "This is because ",
    "To be specific, ",
    "It is well established that ",
    "At its core, ",
    "Put simply, ",
    "Broadly speaking, ",
    "In technical terms, ",
    "By design, ",
    "In essence, ",
    "Based on observed knowledge ",
    "To elaborate on this, ",
];

pub const ABBREVIATIONS: phf::Set<&'static str> = phf_set! {
    "e.g", "i.e", "vs", "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "no", "fig", "cf", "etc", "approx", "dept", "est", "govt", "inc",
    "ltd", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
};

pub const WHY_IS_PREFIX: &str = "why is ";
pub const WHY_ARE_PREFIX: &str = "why are ";
pub const VS_CONNECTOR: &str = " vs ";
pub const AND_CONNECTOR: &str = " and ";
pub const MIN_WORDS_FOR_WHY_PATTERN: usize = 3;
pub const ENTITY_INDEX_WHY_PATTERN: usize = 2;

pub const ARTICLE_A: &str = "a ";
pub const ARTICLE_AN: &str = "an ";
pub const ARTICLE_THE: &str = "the ";

pub const PUNCTUATION_QUESTION: char = '?';
pub const PUNCTUATION_PERIOD: char = '.';
pub const PUNCTUATION_EXCLAMATION: char = '!';

pub const MIN_WORDS_FOR_TRIM_PREDICATE: usize = 3;
pub const MIN_WORDS_FOR_SENTENCE: usize = 8;

/// Form words that follow a creative verb and name the output artifact.
///
/// These are stripped from a prompt like `"write a poem about life"` so only
/// the domain subject (`"life"`) remains as the research topic.
pub const CREATIVE_FORM_WORDS: &[&str] = &[
    "a poem about ",
    "an poem about ",
    "a poem on ",
    "a poem for ",
    "a poem of ",
    "a verse about ",
    "a verse on ",
    "a sonnet about ",
    "a sonnet on ",
    "a haiku about ",
    "a haiku on ",
    "a song about ",
    "a song on ",
    "a story about ",
    "a story on ",
    "a narrative about ",
    "a narrative on ",
    "a tale about ",
    "an essay about ",
    "an essay on ",
    "a report about ",
    "a report on ",
    "an article about ",
    "an article on ",
    "a simple python program ",
    "a python program ",
    "a python script ",
    "a rust program ",
    "a rust function ",
    "a javascript function ",
    "a typescript function ",
    "a go program ",
    "a java program ",
    "a function ",
    "a script ",
    "a program ",
    "a function that ",
    "a function to ",
    "a program that ",
    "a program to ",
    "a script that ",
    "a script to ",
];

/// Connectors that link the form word to the domain subject in a creative prompt.
///
/// After stripping the verb and form word, these connectors are stripped from
/// the leftover so only the core subject remains.
pub const CREATIVE_SUBJECT_CONNECTORS: &[&str] = &[
    "computing ",
    "that computes ",
    "to compute ",
    "for computing ",
    "calculating ",
    "that calculates ",
    "to calculate ",
    "implementing ",
    "that implements ",
    "to implement ",
    "about ",
    "on ",
    "for ",
    "of ",
];

/// Programming language names stripped from the front of a raw subject phrase.
///
/// When a prompt says `"write a simple python hello world program"`, the
/// subject extractor may leave `"python hello world program"` after removing
/// the form word.  Stripping the language name yields `"hello world program"`.
pub const SUBJECT_LANGUAGE_PREFIXES: &[&str] = &[
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

/// Generic adjective modifiers stripped from the front of a raw subject phrase.
///
/// For example `"simple hello world program"` → `"hello world program"`.
pub const SUBJECT_MODIFIER_PREFIXES: &[&str] = &[
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

/// Generic output-type suffixes stripped from the end of a raw subject phrase.
///
/// For example `"hello world program"` → `"hello world"`.
pub const SUBJECT_FORM_SUFFIXES: &[&str] = &[
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

/// Leading article prefix "an ".
pub const ARTICLE_PREFIX_AN: &str = "an ";

/// Leading article prefix "a ".
pub const ARTICLE_PREFIX_A: &str = "a ";

/// Leading article prefix "the ".
pub const ARTICLE_PREFIX_THE: &str = "the ";

/// Minimum word count before significant noun trimming applies.
pub const SIGNIFICANT_NOUN_MIN_WORDS: usize = 3;

/// Minimum character length of the first word for significant noun trimming.
pub const SIGNIFICANT_NOUN_FIRST_WORD_MIN_LEN: usize = 2;

/// Default word count taken when no stop word is found in noun trimming.
pub const SIGNIFICANT_NOUN_FALLBACK_TAKE: usize = 2;

/// Stop words used to delimit significant noun concepts.
pub const SIGNIFICANT_NOUN_STOP_WORDS: &[&str] = &[
    "of",
    "for",
    "in",
    "on",
    "to",
    "from",
    "by",
    "with",
    "and",
    "or",
    "algorithm",
    "example",
    "function",
    "program",
    "method",
    "implementation",
];

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.
