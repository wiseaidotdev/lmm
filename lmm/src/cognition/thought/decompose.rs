// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Goal Decomposer
//!
//! The `GoalDecomposer` expands a user prompt into an ordered list of **sub-goals**
//! based on the detected [`crate::cognition::thought::classifier::ReasoningMode`].
//!
//! ## Sub-goal Derivation
//!
//! Each `ReasoningMode` has a canonical set of **aspect templates**, the natural
//! dimensions along which that type of question should be explored.  The templates
//! are instantiated with the core topic extracted from the prompt, producing a
//! list of concrete sub-goal strings ready for oracle queries.
//!
//! | Mode | Aspect templates |
//! |---|---|
//! | `FactualLookup` | definition, origin, properties, examples, related topics |
//! | `Causal` | mechanism, contributing factors, effects, timeline, prevention |
//! | `Analogical` | shared properties, key differences, context A, context B, evaluation |
//! | `Definitional` | etymology, formal definition, usage, examples, contrast |
//! | `Procedural` | prerequisites, steps, verification, common errors, alternatives |
//! | `Exploratory` | broad overview, notable aspects, history, related concepts, impact |
//! | `Evaluative` | advantages, benefits, disadvantages, pros and cons, when to use |
//! | `TaskExecution` | example program, how to write, code template, getting started, syntax |
//!
//! ## Prioritization
//!
//! After some steps have accumulated [`crate::cognition::thought::belief::BeliefInterval`]
//! scores, `prioritize` re-orders the remaining sub-goals by **uncertainty gap**:
//! sub-goals whose belief interval is widest (most unknown) are promoted to the
//! front so the engine focuses on what it knows least about.
//!
//! ## See Also
//!
//! * [Goal decomposition - Wikipedia](https://en.wikipedia.org/wiki/Decomposition_(computer_science))
//! * [STRIPS planning - Wikipedia](https://en.wikipedia.org/wiki/Stanford_Research_Institute_Problem_Solver)

use crate::cognition::thought::belief::BeliefInterval;
use crate::cognition::thought::classifier::ReasoningMode;
use crate::constants::thought_decompose::{MIN_TOPIC_LENGTH, MIN_WORDS_FOR_TRIM};
use std::cmp::Ordering;

/// Decomposes a prompt into ordered sub-goal strings and re-prioritises them
/// based on accumulated evidence.
///
/// All methods are stateless and pure.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::decompose::GoalDecomposer;
/// use lmm::cognition::thought::classifier::ReasoningMode;
///
/// let goals = GoalDecomposer::decompose("rust programming language", ReasoningMode::Definitional, &[]);
/// assert!(!goals.is_empty());
/// assert!(goals.iter().any(|g| g.contains("rust")));
/// ```
pub struct GoalDecomposer;

impl GoalDecomposer {
    /// Decomposes `prompt` into an ordered list of sub-goal strings.
    ///
    /// The sub-goals are instantiated aspect templates for `mode`, anchored to
    /// the core topic extracted from `prompt`.  When `context` is non-empty it
    /// may supplement the topic extraction.
    ///
    /// # Arguments
    ///
    /// * `prompt`  - Original user input.
    /// * `mode`    - Classified reasoning mode (determines which aspects to explore).
    /// * `context` - Existing knowledge strings that help refine the topic.
    ///
    /// # Time Complexity
    ///
    /// O(|prompt| + |context|)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::decompose::GoalDecomposer;
    /// use lmm::cognition::thought::classifier::ReasoningMode;
    ///
    /// let goals = GoalDecomposer::decompose("why does rust avoid garbage collection", ReasoningMode::Causal, &[]);
    /// assert!(goals.len() >= 3);
    /// ```
    pub fn decompose(prompt: &str, mode: ReasoningMode, context: &[String]) -> Vec<String> {
        let topic = Self::extract_topic(prompt, context);
        if matches!(mode, ReasoningMode::Causal) {
            return Self::decompose_causal(prompt, &topic);
        }
        let aspects = Self::aspects_for_mode(mode);
        aspects
            .iter()
            .map(|aspect| Self::instantiate(aspect, &topic, prompt))
            .collect()
    }

    /// Generates richer causal sub-goals by extracting the full predicate clause
    /// from the prompt rather than just the subject noun.
    ///
    /// For `"why does rust not have a garbage collector?"` this produces aspects
    /// about `"rust memory management"`, `"rust ownership model"`, etc. rather
    /// than the generic `"mechanism behind rust"`.
    ///
    /// # Time Complexity
    ///
    /// O(|prompt|)
    fn decompose_causal(prompt: &str, topic: &str) -> Vec<String> {
        let prompt_lower = prompt.to_ascii_lowercase();
        let clean_prompt = prompt_lower
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim();
        static CAUSAL_OPENERS: &[&str] = &[
            "why does ",
            "why do ",
            "why is ",
            "why are ",
            "why was ",
            "why were ",
            "why would ",
            "why should ",
            "why can't ",
            "why cannot ",
            "why doesn't ",
            "why don't ",
            "how come ",
        ];
        let predicate = CAUSAL_OPENERS
            .iter()
            .find_map(|&pfx| clean_prompt.strip_prefix(pfx))
            .unwrap_or(clean_prompt);
        let predicate_trimmed = predicate.trim();
        let full_question = if predicate_trimmed.is_empty() {
            topic
        } else {
            predicate_trimmed
        };
        let words: Vec<&str> = full_question.split_whitespace().collect();
        let key_phrase = if words.len() > 6 {
            words[..6].join(" ")
        } else {
            full_question.to_string()
        };
        vec![
            format!("{} explanation and rationale", key_phrase),
            format!("why {} design decision", topic),
            format!("{} technical background", key_phrase),
            format!("{} underlying mechanism", topic),
            format!("{} history and motivation", topic),
        ]
    }

    /// Re-orders `sub_goals` so that the most uncertain sub-goals (widest belief
    /// interval or lowest midpoint) appear first.
    ///
    /// `evidenced` maps each sub-goal index to the belief interval accumulated
    /// from the steps that addressed it.  Sub-goals with no evidence are assumed
    /// maximally uncertain and placed first.
    ///
    /// # Arguments
    ///
    /// * `sub_goals` - Full ordered list of sub-goals.
    /// * `evidenced` - Belief intervals indexed in the same order as `sub_goals`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::decompose::GoalDecomposer;
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// let goals = vec!["topic definition".to_string(), "topic origin".to_string()];
    /// let evidence = vec![BeliefInterval::certain(), BeliefInterval::ignorant()];
    /// let prioritized = GoalDecomposer::prioritize(goals, &evidence);
    /// assert_eq!(prioritized[0], "topic origin");
    /// ```
    pub fn prioritize(sub_goals: Vec<String>, evidenced: &[BeliefInterval]) -> Vec<String> {
        let mut indexed: Vec<(String, f64)> = sub_goals
            .into_iter()
            .enumerate()
            .map(|(i, goal)| {
                let uncertainty = evidenced.get(i).map(|b| 1.0 - b.mid()).unwrap_or(1.0);
                (goal, uncertainty)
            })
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.into_iter().map(|(g, _)| g).collect()
    }

    /// Extracts a concise topic phrase from `prompt`, optionally informed by
    /// existing `context` strings.
    ///
    /// Strips leading question words and articles, returning the remaining
    /// content words joined as a phrase.  Falls back to the full `prompt` when
    /// stripping yields nothing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::decompose::GoalDecomposer;
    ///
    /// let t = GoalDecomposer::extract_topic("what is the borrow checker", &[]);
    /// assert_eq!(t, "borrow checker");
    /// ```
    pub fn extract_topic(prompt: &str, _context: &[String]) -> String {
        static FILLER_PREFIXES: &[&str] = &[
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
            "how do i ",
            "how do ",
            "how is ",
            "how are ",
            "how to ",
            "explain ",
            "describe ",
            "define ",
            "tell me about ",
            "why does ",
            "why do ",
            "why is ",
            "why are ",
            "who is ",
            "who was ",
            "where is ",
            "when is ",
            "is it true that ",
            "is ",
            "write a ",
            "write an ",
            "write ",
            "generate a ",
            "generate an ",
            "generate ",
            "create a ",
            "create an ",
            "create ",
            "implement a ",
            "implement an ",
            "implement ",
            "make a ",
            "make an ",
            "code a ",
            "build a ",
            "show me how to ",
            "show me a ",
            "show me ",
            "give me a ",
            "give me an ",
        ];

        let lower = prompt.to_ascii_lowercase();
        let raw = FILLER_PREFIXES
            .iter()
            .find_map(|prefix| lower.strip_prefix(prefix))
            .unwrap_or(prompt.trim());

        let stripped = raw
            .trim_start_matches(|c: char| !c.is_alphanumeric())
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim();

        if stripped.is_empty() {
            return prompt.trim().to_string();
        }
        let core = trim_predicate_decompose(stripped);
        strip_mode_keywords_decompose(&core)
    }

    /// Returns the canonical aspect list for `mode`.
    fn aspects_for_mode(mode: ReasoningMode) -> &'static [&'static str] {
        match mode {
            ReasoningMode::FactualLookup => &[
                "definition of {topic}",
                "origin and history of {topic}",
                "key properties of {topic}",
                "examples of {topic}",
                "related topics to {topic}",
            ],
            ReasoningMode::Causal => &[
                "mechanism behind {topic}",
                "contributing factors to {topic}",
                "effects and consequences of {topic}",
                "timeline of {topic}",
                "how to prevent or address {topic}",
            ],
            ReasoningMode::Analogical => &[
                "shared properties between {topic}",
                "key differences in {topic}",
                "context and background of {topic}",
                "use cases for {topic}",
                "which is better: {topic}",
            ],
            ReasoningMode::Definitional => &[
                "formal definition of {topic}",
                "etymology and origin of {topic}",
                "usage and applications of {topic}",
                "examples illustrating {topic}",
                "contrast: {topic} vs related concepts",
            ],
            ReasoningMode::Procedural => &[
                "prerequisites for {topic}",
                "step by step guide to {topic}",
                "how to verify {topic}",
                "common errors in {topic}",
                "alternatives to {topic}",
            ],
            ReasoningMode::Exploratory => &[
                "{topic} overview",
                "history of {topic}",
                "notable aspects of {topic}",
                "related concepts to {topic}",
                "impact and significance of {topic}",
            ],
            ReasoningMode::Evaluative => &[
                "advantages of {topic}",
                "benefits of {topic}",
                "disadvantages of {topic}",
                "pros and cons of {topic}",
                "when to use {topic}",
            ],
            ReasoningMode::TaskExecution => &[
                "example {topic} program",
                "how to write {topic}",
                "{topic} code template",
                "{topic} getting started",
                "{topic} syntax",
            ],
        }
    }

    /// Instantiates an aspect template by substituting `{topic}` with `topic`.
    ///
    /// When `topic` is empty or generic, falls back to the original `prompt`.
    fn instantiate(aspect: &str, topic: &str, prompt: &str) -> String {
        let used_topic = if topic.len() < MIN_TOPIC_LENGTH {
            prompt.trim()
        } else {
            topic
        };
        aspect.replace("{topic}", used_topic)
    }
}

/// Strips common mode-specific keyword prefixes from a topic left over after
/// question-prefix removal.
///
/// After stripping `"what are the "` from `"what are the advantages of rust"`,
/// the residual is `"advantages of rust"`.  When used as an aspect template
/// anchor this produces redundant text (`"advantages of advantages of rust"`).
/// This function strips those second-level prefixes to yield just `"rust"`.
///
/// # Time Complexity
///
/// O(|s|)
fn strip_mode_keywords_decompose(s: &str) -> String {
    static MODE_PREFIXES: &[&str] = &[
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
        "demerits of ",
        "causes of ",
        "cause of ",
        "reason for ",
        "reasons for ",
        "explanation of ",
        "effect of ",
        "effects of ",
        "definition of ",
        "meaning of ",
        "concept of ",
        "examples of ",
        "example of ",
        "overview of ",
        "history of ",
        "installation of ",
        "setup of ",
        "usage of ",
        "use of ",
        "uses of ",
    ];
    let lower = s.to_ascii_lowercase();
    for prefix in MODE_PREFIXES {
        if lower.starts_with(prefix) {
            let rest = s[prefix.len()..].trim();
            if rest.len() >= 2 {
                return rest.to_string();
            }
        }
    }
    s.to_string()
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Trims a topic string at the first predicate word, keeping only the subject noun phrase.
///
/// When a question prefix is stripped (e.g. "why does "), the remainder may still
/// contain the predicate ("Rust not have a garbage collector").  This function
/// truncates at the first auxiliary, negation, or main-verb word so only the
/// subject noun phrase is returned ("Rust"), producing cleaner search terms.
///
/// Only applies when the input contains 3 or more words to avoid over-truncation.
///
/// # Time Complexity
///
/// O(|s|)
fn trim_predicate_decompose(s: &str) -> String {
    static PREDICATE_WORDS: &[&str] = &[
        "not", "no", "never", "have", "has", "had", "is", "are", "was", "were", "do", "does",
        "did", "will", "would", "could", "should", "can", "cannot", "work", "works", "make",
        "makes", "use", "uses", "need", "needs", "allow", "allows", "prevent", "prevents",
        "require", "requires", "cause", "causes",
    ];
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < MIN_WORDS_FOR_TRIM {
        return s.to_string();
    }
    let cut = words
        .iter()
        .skip(1)
        .position(|w| PREDICATE_WORDS.contains(&w.to_ascii_lowercase().as_str()))
        .map(|i| i + 1)
        .unwrap_or(words.len());
    words[..cut].join(" ")
}
