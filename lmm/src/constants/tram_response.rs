// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # TRAM Response Constants

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
