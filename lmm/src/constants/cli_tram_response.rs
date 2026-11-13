// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # CLI TRAM Response Constants

pub const MIN_QUALITY_WORD_COUNT: usize = 5;
pub const MIN_SENTENCE_WORD_COUNT: usize = 2;
pub const MIN_AVG_WORDS_PER_SENTENCE: usize = 2;
pub const MIN_VERB_COUNT: usize = 1;
pub const MIN_COLD_CORPUS_WORDS: usize = 20;
pub const MIN_SUMMARY_WORDS: usize = 10;
pub const MIN_SUMMARY_SENTENCES: usize = 5;
pub const MAX_SUMMARY_SENTENCES: usize = 8;
pub const SUMMARY_TOP_N: usize = 3;
pub const MIN_TOPIC_WORD_LENGTH: usize = 4;
pub const MAX_PREDICATE_WORDS: usize = 7;
pub const MIN_PARAGRAPH_SENTENCES: usize = 2;
pub const MAX_PARAGRAPH_SENTENCES: usize = 2;
pub const MIN_SENTENCE_LENGTH: usize = 5;
pub const MAX_SENTENCE_LENGTH: usize = 40;
pub const MAX_PARAGRAGH_LENGTH: usize = 180;
pub const DEDUP_THRESHOLD: f64 = 0.65;

pub const VERB_PATTERNS: [&str; 21] = [
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " provides ",
    " enables ",
    " uses ",
    " includes ",
    " supports ",
    " allows ",
    " makes ",
    " creates ",
    " consists ",
    " contains ",
    " designed ",
    " built ",
    " formed ",
    " released ",
    " developed ",
];

pub const ADDITIONAL_VERB_PATTERNS: [&str; 8] = [
    " developed ",
    " began ",
    " defines ",
    " describe ",
    " reason ",
    " causes ",
    " result ",
    " reason",
];

pub const CAUSAL_OPENERS: &[&str] = &[
    "why does ",
    "why do ",
    "why is ",
    "why are ",
    "why was ",
    "why would ",
    "why should ",
    "why can't ",
    "why cannot ",
    "why doesn't ",
    "why don't ",
    "how come ",
];

pub const VS_SPLITTERS: &[&str] = &[" vs ", " versus "];

pub const SITE_NAMES: &[&str] = &[
    "geeksforgeeks",
    "stackoverflow",
    "reddit",
    "medium",
    "github",
    "wikipedia",
    "quora",
    "hackernews",
    "dev.to",
    "towardsdatascience",
    "freecodecamp",
    "programiz",
    "tutorialspoint",
];

pub const DANGLING_PREPOSITIONS: &[&str] = &[
    "in", "at", "of", "to", "for", "on", "by", "with", "from", "into", "onto", "upon", "about",
    "over", "under", "between", "among", "through", "within", "without", "during", "and", "or",
    "but", "the", "a", "an",
];

pub const STOP_WORDS: [&str; 48] = [
    "what",
    "does",
    "how",
    "the",
    "and",
    "for",
    "with",
    "about",
    "install",
    "make",
    "build",
    "create",
    "setup",
    "configure",
    "learn",
    "understand",
    "explain",
    "describe",
    "find",
    "show",
    "tell",
    "give",
    "help",
    "work",
    "works",
    "working",
    "used",
    "using",
    "uses",
    "need",
    "needs",
    "want",
    "that",
    "this",
    "into",
    "from",
    "have",
    "been",
    "will",
    "would",
    "could",
    "should",
    "when",
    "where",
    "which",
    "who",
    "why",
    "also",
];

pub const ANALOGICAL_VS_SPLITTERS: &[&str] = &[" vs ", " versus ", " compared to "];

pub const CONNECTORS: &[&str] = &[
    "On the other hand,",
    "By contrast,",
    "Meanwhile,",
    "In comparison,",
    "Conversely,",
];

pub const CAUSAL_PREFIXES: &[(&str, &str)] = &[
    ("why does ", "does"),
    ("why do ", "do"),
    ("why is ", "is"),
    ("why are ", "are"),
    ("why was ", "was"),
    ("why were ", "were"),
    ("why would ", "would"),
    ("why should ", "should"),
    ("why can't ", "cannot"),
    ("why cannot ", "cannot"),
    ("why doesn't ", "does not"),
    ("why don't ", "do not"),
    ("how come ", ""),
];

pub const AUX_VERBS: [&str; 23] = [
    "is", "are", "was", "were", "has", "have", "had", "does", "do", "did", "will", "would",
    "could", "should", "can", "cannot", "doesn't", "don't", "can't", "won't", "wouldn't",
    "couldn't", "not",
];

pub const PREDICATE_STARTERS: &[&str] = &[
    "not", "no", "never", "have", "has", "had", "make", "makes", "work", "works", "allow",
    "allows", "prevent", "prevents", "require", "requires", "cause", "causes", "use", "uses",
    "need", "needs",
];
