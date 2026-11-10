// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Expert System: Rule Representation
//!
//! This module implements **IF-THEN production rules**.
//!
//! Rules are the atomic units of an expert system's knowledge base.  Each rule
//! has the form:
//!
//! ```text
//! IF condition_1 AND condition_2 AND ... THEN conclusion [certainty_factor]
//! ```
//!
//! Rules here are **derived automatically at runtime** from ingested
//! [`crate::cognition::knowledge::DocumentChunk`]s via lightweight relational
//! pattern matching, so they always reflect the topic being queried.
//!
//! ## Condition extraction heuristics
//!
//! The extractor scans each chunk for SVO-like (Subject-Verb-Object) triples
//! using a small set of relation verbs:
//!
//! | Pattern | Predicate |
//! |---|---|
//! | `X is Y`         | `is` |
//! | `X are Y`        | `are` |
//! | `X enables Y`    | `enables` |
//! | `X provides Y`   | `provides` |
//! | `X requires Y`   | `requires` |
//! | `X supports Y`   | `supports` |
//! | `X uses Y`       | `uses` |
//! | `X prevents Y`   | `prevents` |
//! | `X guarantees Y` | `guarantees` |
//! | `X has Y`        | `has` |

use std::collections::{HashMap, HashSet};

/// A set of relation verbs used to extract subject-predicate-object triples.
static RELATION_VERBS: &[&str] = &[
    "is",
    "are",
    "enables",
    "provides",
    "requires",
    "supports",
    "uses",
    "prevents",
    "guarantees",
    "has",
    "ensures",
    "allows",
    "contains",
    "includes",
    "defines",
    "represents",
    "describes",
    "produces",
];

/// An atomic condition in a rule's antecedent.
///
/// Models a simple subject-predicate-object triple.  The condition is
/// satisfied when the knowledge base contains a matching assertion.
///
/// # Examples
///
/// ```rust
/// use lmm::expert::rule::Condition;
/// let c = Condition::new("rust", "is", "fast", false);
/// assert!(!c.negated);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    /// Subject token (e.g. "rust", "python").
    pub subject: String,
    /// Predicate verb (e.g. "is", "enables", "requires").
    pub predicate: String,
    /// Object token (e.g. "fast", "memory-safe", "garbage-collection").
    pub object: String,
    /// When `true` the condition asserts the **absence** of the triple.
    pub negated: bool,
}

impl Condition {
    /// Creates a new `Condition`.
    ///
    /// # Arguments
    ///
    /// * `subject`   - The subject word.
    /// * `predicate` - The relation verb.
    /// * `object`    - The object word or phrase.
    /// * `negated`   - Whether the condition is negated.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        negated: bool,
    ) -> Self {
        Self {
            subject: subject.into().to_ascii_lowercase(),
            predicate: predicate.into().to_ascii_lowercase(),
            object: object.into().to_ascii_lowercase(),
            negated,
        }
    }

    /// Returns the condition as a human-readable string, deduplicating
    /// any adjacent repeated words that may arise from imperfect sentence parsing.
    ///
    /// # Time Complexity
    ///
    /// O(|subject| + |predicate| + |object|)
    pub fn to_text(&self) -> String {
        let raw = if self.negated {
            format!(
                "{} does not {} {}",
                self.subject, self.predicate, self.object
            )
        } else {
            format!("{} {} {}", self.subject, self.predicate, self.object)
        };
        dedup_words(&raw)
    }
}

/// A production rule: IF all conditions hold THEN conclusion follows with CF.
///
///
/// ```text
/// rule_id : if Condition_1 and Condition_2 ... then Conclusion [CF].
/// ```
///
/// # Examples
///
/// ```rust
/// use lmm::expert::rule::{Rule, Condition};
/// let r = Rule {
///     id: "rust_safe".to_string(),
///     conditions: vec![Condition::new("rust", "has", "ownership", false)],
///     conclusion: "rust is memory-safe".to_string(),
///     certainty_factor: 0.85,
/// };
/// assert_eq!(r.id, "rust_safe");
/// ```
#[derive(Debug, Clone)]
pub struct Rule {
    /// Unique identifier (auto-generated from subject + predicate tokens).
    pub id: String,
    /// Antecedent conditions; all must hold (AND semantics).
    pub conditions: Vec<Condition>,
    /// The derived assertion when all conditions hold.
    pub conclusion: String,
    /// Prospector-style certainty factor ∈ [0.0, 1.0].
    pub certainty_factor: f64,
}

/// A collection of IF-THEN rules derived from an ingested knowledge base.
///
/// The rule base is rebuilt each time new knowledge is ingested; it is not
/// persisted across runs, making it fully topic-adaptive.
///
/// # Time Complexity of construction
///
/// O(|chunks| × |tokens_per_chunk|)
#[derive(Debug, Default, Clone)]
pub struct RuleBase {
    /// The extracted production rules.
    pub rules: Vec<Rule>,
    /// Simple ground-truth facts (subject-predicate-object) for condition checking.
    pub facts: Vec<Condition>,
}

impl RuleBase {
    /// Creates an empty `RuleBase`.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn new() -> Self {
        Self::default()
    }

    /// Derives rules and facts from a slice of knowledge-base text chunks.
    ///
    /// For each chunk, a lightweight SVO extractor identifies relation verbs and
    /// creates one rule per relation found.  The certainty factor is proportional
    /// to the number of relation verbs found in the chunk (more evidence → higher CF).
    ///
    /// # Arguments
    ///
    /// * `chunks` - Slice of raw text strings from `KnowledgeIndex::query`.
    ///
    /// # Time Complexity
    ///
    /// O(|chunks| × |tokens_per_chunk|)
    ///
    /// # Space Complexity
    ///
    /// O(|rules| + |facts|) where |rules| ≤ |chunks| × |RELATION_VERBS|
    pub fn from_chunks(chunks: &[&str]) -> Self {
        let mut base = Self::new();
        let mut seen_conclusions: HashMap<String, bool> = HashMap::new();
        let mut rule_counter = 0usize;

        for chunk in chunks {
            let triples = extract_triples(chunk);
            if triples.is_empty() {
                continue;
            }

            let rel_count = triples.len() as f64;
            let base_cf = (rel_count / 5.0).clamp(0.1, 1.0);

            for triple in &triples {
                let conclusion = triple.to_text();
                let conclusion = dedup_words(&conclusion);
                if seen_conclusions.contains_key(&conclusion) {
                    continue;
                }
                seen_conclusions.insert(conclusion.clone(), true);

                let id = format!("rule_{}", rule_counter);
                rule_counter += 1;

                let rule = Rule {
                    id,
                    conditions: vec![triple.clone()],
                    conclusion,
                    certainty_factor: base_cf.min(0.95),
                };
                base.facts.push(triple.clone());
                base.rules.push(rule);
            }
        }

        base
    }

    /// Returns all rules whose conclusion contains any token from `goal`.
    ///
    /// # Arguments
    ///
    /// * `goal` - Natural-language goal string (e.g. "rust is fast").
    ///
    /// # Time Complexity
    ///
    /// O(|rules| × |goal_tokens|)
    pub fn matching_rules(&self, goal: &str) -> Vec<&Rule> {
        let goal_tokens: Vec<String> = goal
            .split_whitespace()
            .map(|w| w.to_ascii_lowercase())
            .filter(|w| w.len() >= 3)
            .collect();

        self.rules
            .iter()
            .filter(|r| {
                let conc_lower = r.conclusion.to_ascii_lowercase();
                goal_tokens.iter().any(|t| conc_lower.contains(t.as_str()))
            })
            .collect()
    }

    /// Returns `true` if the given condition is satisfied by the fact base.
    ///
    /// # Arguments
    ///
    /// * `cond` - The condition to check.
    ///
    /// # Time Complexity
    ///
    /// O(|facts|)
    pub fn condition_holds(&self, cond: &Condition) -> bool {
        let found = self.facts.iter().any(|f| {
            f.subject.contains(&cond.subject)
                && f.predicate == cond.predicate
                && f.object.contains(&cond.object)
        });
        if cond.negated { !found } else { found }
    }
}

/// Extracts SVO triples from a text chunk using RELATION_VERBS as pivot words.
///
/// For each relation verb found, the left tokens (up to 3) form the subject
/// and the right tokens (up to 4) form the object.
///
/// # Time Complexity
///
/// O(|text| × |RELATION_VERBS|)
fn extract_triples(text: &str) -> Vec<Condition> {
    let words: Vec<&str> = text
        .split(['.', ',', ';', ':'])
        .flat_map(|clause| clause.split_whitespace())
        .collect();

    let mut triples = Vec::new();

    for (i, word) in words.iter().enumerate() {
        let lower = word.to_ascii_lowercase();
        let verb = lower.trim_matches(|c: char| !c.is_ascii_alphabetic());

        if !RELATION_VERBS.contains(&verb) {
            continue;
        }

        let raw_subject_words: Vec<String> = words[i.saturating_sub(3)..i]
            .iter()
            .filter(|w| w.len() >= 3 && !is_stop(w))
            .map(|w| w.to_ascii_lowercase())
            .collect();

        let mut seen_subj = HashSet::new();
        let subject: String = raw_subject_words
            .into_iter()
            .filter(|w| seen_subj.insert(w.clone()))
            .collect::<Vec<_>>()
            .join(" ");

        let obj_end = (i + 5).min(words.len());
        let object: String = words[i + 1..obj_end]
            .iter()
            .filter(|w| w.len() >= 2 && !is_stop(w))
            .map(|w| {
                w.to_ascii_lowercase()
                    .trim_matches(|c: char| !c.is_ascii_alphabetic())
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join(" ");

        if subject.is_empty() || object.is_empty() {
            continue;
        }

        triples.push(Condition::new(subject, verb, object, false));
    }

    triples
}

/// Removes adjacent duplicate words from a string (case-insensitive comparison).
///
/// Prevents conclusions like "rust rust is multi-paradigm" that arise when two
/// consecutive source sentences are merged and the same noun appears twice.
///
/// # Time Complexity
///
/// O(|text|)
fn dedup_words(text: &str) -> String {
    let mut prev = "";
    text.split_whitespace()
        .filter(|w| {
            if w.eq_ignore_ascii_case(prev) {
                false
            } else {
                prev = w;
                true
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns `true` for common stop words that add no meaning to a triple.
///
/// # Time Complexity
///
/// O(1): fixed set comparison.
fn is_stop(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "the"
            | "a"
            | "an"
            | "and"
            | "or"
            | "of"
            | "in"
            | "to"
            | "it"
            | "its"
            | "this"
            | "that"
            | "for"
            | "with"
            | "by"
            | "at"
            | "on"
            | "as"
            | "be"
            | "been"
            | "can"
            | "also"
            | "not"
            | "but"
            | "from"
            | "into"
            | "than"
            | "when"
            | "which"
            | "who"
            | "how"
            | "what"
            | "where"
    )
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
