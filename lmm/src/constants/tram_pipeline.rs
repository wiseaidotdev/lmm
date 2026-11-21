// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # TRAM Pipeline Constants

pub const ORACLE_CACHE_SIZE: usize = 5;
pub const KB_QUERY_TOP_K: usize = 8;
pub const MIN_TOPIC_WORD_LENGTH: usize = 4;
pub const MIN_COLD_CORPUS_WORDS: usize = 20;
pub const MIN_SENTENCE_WORDS_FOR_PIPELINE: usize = 6;

/// Minimum word count for an answer to earn a non-zero quality reward.
pub const REWARD_MIN_ANSWER_WORDS: usize = 20;

/// Reward bonus applied when the answer appears to contain novel content
/// not literally copied from the top KB chunk.
pub const REWARD_NOVEL_CONTENT_BONUS: f64 = 0.3;

/// Weight applied to KB topic coverage when computing the episode reward.
pub const REWARD_KB_COVERAGE_WEIGHT: f64 = 0.5;

/// Weight applied to prompt-relevance when computing the episode reward.
pub const REWARD_PROMPT_RELEVANCE_WEIGHT: f64 = 0.5;

/// Default confidence assigned to a skill synthesised from KB content.
pub const SKILL_SYNTHESISED_CONFIDENCE: f64 = 0.7;

/// Number of KB chunks to retrieve when synthesising a new skill's content.
pub const SKILL_SYNTHESIS_TOP_K: usize = 5;

/// LCG multiplier used by the Fisher-Yates shuffle in `assemble_text`.
pub const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// LCG addend used by the Fisher-Yates shuffle in `assemble_text`.
pub const LCG_ADDEND: u64 = 1_442_695_040_888_963_407;

/// Right-shift applied to the LCG state to derive the swap index.
pub const LCG_SHIFT: u32 = 33;

/// Minimum paragraph count for `ParagraphShaper` in `assemble_text`.
pub const SHAPER_MIN_PARAGRAPHS: usize = 2;

/// Maximum paragraph count for `ParagraphShaper` in `assemble_text`.
pub const SHAPER_MAX_PARAGRAPHS: usize = 2;

/// Minimum sentences per paragraph for `ParagraphShaper`.
pub const SHAPER_MIN_SENTENCES: usize = 5;

/// Minimum sentence word count for `ParagraphShaper`.
pub const SHAPER_MIN_SENTENCE_WORDS: usize = 40;

/// Maximum sentence word count for `ParagraphShaper`.
pub const SHAPER_MAX_SENTENCE_WORDS: usize = 180;

/// Discourse connector prefixes stripped before sentence assembly.
///
/// These transition phrases are removed from the beginning of sentences so
/// that the assembled paragraph does not read like a concatenation of
/// independently-framed paragraphs.
pub const DISCOURSE_CONNECTORS: &[&str] = &[
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

/// Short words excluded when picking the representative topic word.
pub const TOPIC_SKIP_WORDS: &[&str] = &[
    "what", "does", "how", "the", "and", "for", "with", "about", "that", "this", "from", "into",
    "over",
];

/// Fallback phrases appended to the topic when no KB content is found.
pub const FALLBACK_PHRASES: &[&str] = &[
    "is a foundational topic with many interconnected aspects.",
    "spans many dimensions of knowledge and ongoing research.",
    "continues to be studied and discussed across disciplines.",
    "has shaped thought and practice in significant ways.",
    "is widely explored across academic and professional fields.",
];

/// Minimum word count for `enhanced_text` to be treated as substantive creative
/// content. Below this threshold the pipeline falls back to the creative hint.
pub const MIN_CREATIVE_CONTENT_WORDS: usize = 20;

/// Hint returned when the pipeline cannot assemble a poem from the KB.
pub const CREATIVE_HINT_POEM: &str = "A {topic} could not be composed from the available knowledge base. \
     Try enabling deep mode for richer results, or provide a knowledge file about the subject.";

/// Hint returned when the pipeline cannot assemble a story from the KB.
pub const CREATIVE_HINT_STORY: &str = "A {topic} could not be assembled from the available knowledge base. \
     Try enabling deep mode for richer results, or provide a knowledge file about the subject.";

/// Hint returned when the pipeline cannot assemble an essay from the KB.
pub const CREATIVE_HINT_ESSAY: &str = "An {topic} could not be composed from the available knowledge base. \
     Try enabling deep mode for richer results, or provide a knowledge file about the subject.";

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.
