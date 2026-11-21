// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # TRAM Response Constants

/// Opening prefix for causal mode when the topic is a short noun.
pub const FRAMER_CAUSAL_SHORT_PREFIX: &str = "The reason";

/// Fallback opener for causal mode when the answer lacks causal content.
pub const FRAMER_CAUSAL_WHAT_KNOWN_PREFIX: &str = "Here is what is known about";

/// Suffix appended after the topic in a predicate-rich causal opener.
pub const FRAMER_CAUSAL_BECAUSE_SUFFIX: &str = "because";

/// Suffix that follows two entity names in an analogical opener.
pub const FRAMER_ANALOGICAL_KEY_DIFFERENCES: &str = "the key differences are as follows.";

/// Fallback analogical opener when no vs/and split is found.
pub const FRAMER_ANALOGICAL_COMPARING_SUFFIX: &str = "Comparing";

/// Prefix for definitional mode openers.
pub const FRAMER_DEFINITIONAL_PREFIX: &str = "By definition,";

/// Prefix for procedural mode openers.
pub const FRAMER_PROCEDURAL_PREFIX: &str = "Here is how to";

/// Label used in evaluative openers for inconvenience prompts.
pub const FRAMER_EVALUATIVE_INCONVENIENCES: &str = "inconveniences";

/// Label used in evaluative openers for limitation prompts.
pub const FRAMER_EVALUATIVE_LIMITATIONS: &str = "limitations";

/// Label used in evaluative openers for disadvantage prompts.
pub const FRAMER_EVALUATIVE_DISADVANTAGES: &str = "disadvantages";

/// Label used in evaluative openers for drawback prompts.
pub const FRAMER_EVALUATIVE_DRAWBACKS: &str = "drawbacks";

/// Label used in evaluative openers for weakness prompts.
pub const FRAMER_EVALUATIVE_WEAKNESSES: &str = "weaknesses";

/// Label used in evaluative openers for pros-and-cons prompts.
pub const FRAMER_EVALUATIVE_PROS_CONS: &str = "pros and cons";

/// Label used in evaluative openers for advantage/benefit prompts.
pub const FRAMER_EVALUATIVE_ADVANTAGES: &str = "advantages";

/// Prefix for task-execution mode openers.
pub const FRAMER_TASK_EXECUTION_PREFIX: &str = "Here is the solution for";

/// Evaluative opener suffix introducing an enumeration.
pub const FRAMER_EVAL_SUFFIX_AS_FOLLOWS: &str = "are as follows:";

/// Evaluative opener fragment signalling notable qualities.
pub const FRAMER_EVAL_SUFFIX_NOTABLE: &str = "offers several notable";

/// Evaluative opener suffix for the "stand out" list-introduction pattern.
///
/// Used in Template 2 of `build_evaluative_opener`:
/// `"{marker}, the following {label} of {topic} stand out:"`
pub const FRAMER_EVAL_SUFFIX_STAND_OUT: &str = "stand out:";

/// Evaluative opener suffix inviting closer inspection.
pub const FRAMER_EVAL_SUFFIX_EXAMINATION: &str = "merit close examination:";

/// Evaluative opener fragment for the reveal pattern.
pub const FRAMER_EVAL_SUFFIX_REVEALS: &str = "reveals several";

/// Words in an IA abstract sentence that signal an evaluative point (a feature,
/// advantage, or capability worth highlighting in evaluative responses).
///
/// Only action verbs and adjectives that imply a **user-visible benefit** are
/// included.  Generic subject-verb phrases like `"emphasizes"` or `"provides"`
/// that describe the subject's own design rather than user benefit have been
/// removed to prevent definitional/historical sentences from leaking through.
///
/// When the pipeline lacks rich KB content for evaluative prompts, any IA
/// abstract sentence containing one of these words is extracted as an
/// advantage-point rather than discarding the whole abstract as a definition.
pub const EVALUATIVE_SIGNAL_WORDS: &[&str] = &[
    "ensures",
    "enables",
    "allows",
    "supports",
    "offers",
    "features",
    "without",
    "eliminates",
    "prevents",
    "guarantees",
    "improves",
    "reduces",
    "increases",
    "alternative",
    "safe",
    "fast",
    "efficient",
    "reliable",
    "concurrent",
    "parallel",
    "zero-cost",
    "compile-time",
    "performance",
    "memory safety",
    "type safety",
    "concurrency",
    "abstraction",
    "advantage",
    "benefit",
    "no garbage",
    "fearless",
    "ownership",
    "borrow",
    "trait",
];

/// Sentence-level exclusion patterns for [`EVALUATIVE_SIGNAL_WORDS`] filtering.
///
/// A sentence that matches any exclusion pattern is dropped even when it
/// contains a signal word.  This prevents historical, attribution, or
/// definitional sentences from passing the evaluative filter.
///
/// Examples of sentences blocked by this list:
/// - `"Software developer Graydon Hoare created Rust in 2006 while working at Mozilla"`
/// - `"The first stable release, Rust 1.0, was published in May 2015"`
/// - `"The project has been sponsored by the Rust Foundation since February 2021"`
/// - `"Rust is a general-purpose programming language that emphasizes..."`
/// - `"Python is a high-level, general-purpose programming language..."`
pub const EVALUATIVE_EXCLUSION_PATTERNS: &[&str] = &[
    "was created",
    "was published",
    "was founded",
    "in 20",
    "sponsored by",
    "working at",
    "first stable",
    "officially sponsored",
    "while working",
    "created by",
    "developed by",
    "designed by",
    "introduced in",
    "released in",
    "since 20",
    "is a general-purpose",
    "is a high-level",
    "is a systems programming",
    "is a statically typed",
    "is a dynamically typed",
    "is an object-oriented",
    "is a multi-paradigm",
    "is a compiled",
    "is a scripting",
    "is an interpreted",
    "is a low-level",
    "is an open-source",
];

/// Sequence terminating the evaluative opener phrase in a framed response.
///
/// Everything up to and including this string is protected from stochastic
/// enhancement so the opening sentence is never mangled by word substitution.
/// [`crate::cognition::response_framer::ResponseFramer::frame`] uses `":\n\n"`
/// as the separator for evaluative mode so this constant matches that boundary.
pub const RESPONSE_OPENER_TERMINATOR: &str = ":\n\n";

/// Minimum number of sentences the evaluative filter must retain before the
/// result is accepted. Fewer sentences than this causes a fallback to
/// historical-only stripping without signal-word gating.
pub const MIN_EVALUATIVE_SENTENCES: usize = 2;

pub const MIN_RESPONSE_LENGTH: usize = 30;
pub const MAX_REPETITION_THRESHOLD: f64 = 0.4;
pub const MIN_SENTENCE_LENGTH: usize = 5;
pub const MIN_SENTENCE_VARIETY: usize = 3;
pub const MIN_WORDS_FOR_VALID_SUMMARY: usize = 8;
pub const MIN_SENTENCE_LEN_FOR_VALID_SUMMARY: usize = 3;
pub const MIN_VERB_COUNT_FOR_VALID_SUMMARY: usize = 1;

pub const RELEVANCE_KEYWORDS: [&str; 7] = [
    "related",
    "relevant",
    "pertinent",
    "connected",
    "applicable",
    "associated",
    "linked",
];

pub const IRRELEVANCE_KEYWORDS: [&str; 6] = [
    "unrelated",
    "irrelevant",
    "disconnected",
    "inapplicable",
    "foreign",
    "extraneous",
];

pub const QUALITY_KEYWORDS: [&str; 8] = [
    "comprehensive",
    "thorough",
    "detailed",
    "complete",
    "accurate",
    "precise",
    "informative",
    "substantial",
];

pub const POOR_QUALITY_KEYWORDS: [&str; 8] = [
    "brief",
    "incomplete",
    "limited",
    "superficial",
    "shallow",
    "inadequate",
    "fragmented",
    "minimal",
];

pub const REFUSAL_PATTERNS: [&str; 5] = [
    "i don't know",
    "i cannot",
    "i'm not sure",
    "i am not sure",
    "unknown",
];

pub const LOW_CONTENT_PATTERNS: [&str; 4] =
    ["title only", "no information", "insufficient", "not enough"];

pub const RELEVANCE_INTENT_PATTERNS: [&str; 6] = [
    "what is", "explain", "describe", "how does", "why is", "define",
];

pub const INTENT_KEYWORD_MAP: [(&str, [&str; 5]); 3] = [
    (
        "advantages",
        ["benefit", "pro", "strength", "merit", "gain"],
    ),
    (
        "disadvantages",
        ["drawback", "con", "weakness", "limitation", "flaw"],
    ),
    (
        "comparison",
        ["versus", "vs", "against", "compared", "difference"],
    ),
];

pub const STRUCTURAL_MARKERS: [&str; 8] = [
    "first",
    "second",
    "third",
    "finally",
    "moreover",
    "furthermore",
    "however",
    "therefore",
];

pub const ANALOGY_PATTERNS: [&str; 5] = [
    "like a",
    "similar to",
    "analogous to",
    "resembles",
    "comparable to",
];

pub const COMPARISON_PATTERNS: [&str; 5] = [
    "compared to",
    "in contrast to",
    "unlike",
    "whereas",
    "while",
];

pub const PARAGRAPH_CONNECTORS: [&str; 10] = [
    "furthermore",
    "in addition",
    "moreover",
    "additionally",
    "besides",
    "also",
    "moreover",
    "consequently",
    "as a result",
    "therefore",
];

pub const PROSE_CONNECTORS: [&str; 15] = [
    "furthermore",
    "in addition",
    "moreover",
    "additionally",
    "consequently",
    "as a result",
    "therefore",
    "thus",
    "hence",
    "however",
    "nevertheless",
    "in contrast",
    "notably",
    "significantly",
    "interestingly",
];

pub const SHAPER_CONNECTORS: [&str; 8] = [
    "in practice",
    "more specifically",
    "on the technical side",
    "from a practical standpoint",
    "in this context",
    "to elaborate",
    "to be specific",
    "building on this",
];

pub const VERB_PATTERNS: [&str; 20] = [
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
];

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
