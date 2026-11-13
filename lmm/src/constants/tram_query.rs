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
