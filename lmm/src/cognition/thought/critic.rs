// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Internal Critic
//!
//! The `InternalCritic` monitors an in-progress [`crate::cognition::thought::ThoughtChain`]
//! for three classes of reasoning failure and returns a [`CritiqueResult`] when
//! one is detected, so the [`crate::cognition::thought::AdaptiveThoughtEngine`]
//! can insert a corrective step and pivot its strategy.
//!
//! ## Failure Classes
//!
//! | Class | Detection | Correction Signal |
//! |---|---|---|
//! | **Contradiction** | Two conclusions assert conflicting facts about the same subject | `CritiqueResult::Contradiction` |
//! | **Stagnation** | Belief midpoint has not improved by ε across `patience` steps | `CritiqueResult::StagnantBelief` |
//! | **Weak premise** | A step's premise is too short or its belief interval is unresolved | `CritiqueResult::WeakPremise` |
//!
//! ## Contradiction Detection
//!
//! Two conclusions `A` and `B` are considered contradictory when:
//! 1. Their token overlap exceeds `0.5` (they discuss the same subject), **and**
//! 2. One contains a negation marker (`not`, `no`, `never`, `without`, `cannot`,
//!    `doesn't`, `isn't`, `aren't`, `won't`) that the other lacks.
//!
//! ## See Also
//!
//! * [Paraconsistent logic - Wikipedia](https://en.wikipedia.org/wiki/Paraconsistent_logic)
//! * [Metacognition - Wikipedia](https://en.wikipedia.org/wiki/Metacognition)

use crate::cognition::thought::belief::BeliefInterval;
use std::collections::HashSet;

/// A specific reasoning failure detected by the [`InternalCritic`].
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::critic::CritiqueResult;
///
/// let result = CritiqueResult::Contradiction { step_a: 0, step_b: 2 };
/// assert!(result.is_critical());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum CritiqueResult {
    /// Two steps make conflicting claims about the same subject.
    Contradiction {
        /// Index of the earlier step.
        step_a: usize,
        /// Index of the later step.
        step_b: usize,
    },
    /// Belief has not improved meaningfully over several consecutive steps.
    StagnantBelief {
        /// Number of consecutive stagnant steps detected.
        stall_count: usize,
    },
    /// A step's premise is too weak to support reliable reasoning.
    WeakPremise {
        /// Index of the offending step.
        step: usize,
        /// A suggested corrective action in prose.
        suggestion: String,
    },
}

impl CritiqueResult {
    /// Returns `true` for critiques that require an immediate corrective pivot.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::critic::CritiqueResult;
    ///
    /// assert!(CritiqueResult::Contradiction { step_a: 0, step_b: 1 }.is_critical());
    /// assert!(CritiqueResult::StagnantBelief { stall_count: 3 }.is_critical());
    /// assert!(CritiqueResult::WeakPremise { step: 0, suggestion: String::new() }.is_critical());
    /// ```
    pub fn is_critical(&self) -> bool {
        true
    }

    /// Returns a human-readable description of the detected failure.
    pub fn describe(&self) -> String {
        match self {
            Self::Contradiction { step_a, step_b } => format!(
                "Steps {} and {} assert contradictory conclusions about the same subject.",
                step_a, step_b
            ),
            Self::StagnantBelief { stall_count } => format!(
                "Belief has not improved over {} consecutive steps, pivoting strategy.",
                stall_count
            ),
            Self::WeakPremise { step, suggestion } => {
                format!(
                    "Step {} has a weak premise. Suggestion: {}",
                    step, suggestion
                )
            }
        }
    }
}

/// Minimal snapshot of a reasoning step passed to the critic.
///
/// The critic works with this lightweight view rather than the full
/// `ThoughtStep` to avoid circular dependencies between sub-modules.
#[derive(Debug, Clone)]
pub struct StepView<'a> {
    /// Step index.
    pub index: usize,
    /// The step's premise text.
    pub premise: &'a str,
    /// The step's conclusion text.
    pub conclusion: &'a str,
    /// The step's belief interval at the time it was recorded.
    pub belief: BeliefInterval,
}

/// Monitors a reasoning chain and flags the first detected failure.
///
/// Stateless; each call to [`InternalCritic::evaluate`] scans the provided
/// step slice from scratch.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::critic::{InternalCritic, StepView};
/// use lmm::cognition::thought::belief::BeliefInterval;
///
/// let critic = InternalCritic::new(0.02, 3);
/// let steps = vec![
///     StepView { index: 0, premise: "X is true", conclusion: "Y follows", belief: BeliefInterval::new(0.6, 0.9) },
///     StepView { index: 1, premise: "Y follows", conclusion: "Z is confirmed", belief: BeliefInterval::new(0.7, 0.9) },
/// ];
/// assert!(critic.evaluate(&steps).is_none());
/// ```
pub struct InternalCritic {
    /// Minimum belief midpoint improvement per step to avoid stagnation.
    pub improvement_epsilon: f64,
    /// Number of non-improving steps before `StagnantBelief` is raised.
    pub stall_patience: usize,
}

impl InternalCritic {
    /// Constructs a new `InternalCritic`.
    ///
    /// # Arguments
    ///
    /// * `improvement_epsilon` - Minimum belief midpoint delta per step to avoid stagnation.
    /// * `stall_patience`      - Consecutive non-improving steps before raising stagnation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::critic::InternalCritic;
    ///
    /// let critic = InternalCritic::new(0.02, 3);
    /// assert_eq!(critic.stall_patience, 3);
    /// ```
    pub fn new(improvement_epsilon: f64, stall_patience: usize) -> Self {
        Self {
            improvement_epsilon: improvement_epsilon.clamp(0.0, 1.0),
            stall_patience: stall_patience.max(1),
        }
    }

    /// Scans `steps` for the first reasoning failure and returns it, or `None`
    /// when the chain is coherent so far.
    ///
    /// Checks are applied in order: contradiction → stagnation → weak premise.
    /// The first detected failure is returned immediately.
    ///
    /// # Arguments
    ///
    /// * `steps` - Ordered slice of step snapshots representing the chain so far.
    ///
    /// # Time Complexity
    ///
    /// O(|steps|²) for contradiction detection; O(|steps|) for stagnation and
    /// weak premise checks.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::critic::{InternalCritic, StepView, CritiqueResult};
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// let critic = InternalCritic::new(0.05, 2);
    ///
    /// let flat = vec![
    ///     StepView { index: 0, premise: "some premise text here", conclusion: "first conclusion text", belief: BeliefInterval::new(0.5, 0.8) },
    ///     StepView { index: 1, premise: "same premise text here", conclusion: "first conclusion text", belief: BeliefInterval::new(0.51, 0.81) },
    ///     StepView { index: 2, premise: "same premise text here", conclusion: "first conclusion text", belief: BeliefInterval::new(0.52, 0.82) },
    /// ];
    /// assert!(critic.evaluate(&flat).is_some());
    /// ```
    pub fn evaluate<'a>(&self, steps: &[StepView<'a>]) -> Option<CritiqueResult> {
        if let Some(c) = self.detect_contradiction(steps) {
            return Some(c);
        }
        if let Some(s) = self.detect_stagnation(steps) {
            return Some(s);
        }
        self.detect_weak_premise(steps)
    }

    /// Checks all step-pair combinations for contradictory conclusions.
    fn detect_contradiction<'a>(&self, steps: &[StepView<'a>]) -> Option<CritiqueResult> {
        static NEGATION_MARKERS: &[&str] = &[
            "not", "no", "never", "without", "cannot", "doesn't", "isn't", "aren't", "won't",
            "can't", "neither", "nor",
        ];

        for i in 0..steps.len() {
            for j in (i + 1)..steps.len() {
                let ca = steps[i].conclusion;
                let cb = steps[j].conclusion;

                if !Self::high_token_overlap(ca, cb) {
                    continue;
                }

                let neg_a = Self::contains_negation(ca, NEGATION_MARKERS);
                let neg_b = Self::contains_negation(cb, NEGATION_MARKERS);

                if neg_a != neg_b {
                    return Some(CritiqueResult::Contradiction {
                        step_a: steps[i].index,
                        step_b: steps[j].index,
                    });
                }
            }
        }
        None
    }

    /// Checks whether belief midpoint has improved over the last `patience` steps.
    fn detect_stagnation<'a>(&self, steps: &[StepView<'a>]) -> Option<CritiqueResult> {
        if steps.len() < self.stall_patience + 1 {
            return None;
        }

        let tail = &steps[steps.len() - self.stall_patience..];
        let first_mid = tail.first().map(|s| s.belief.mid()).unwrap_or(0.0);
        let last_mid = tail.last().map(|s| s.belief.mid()).unwrap_or(0.0);

        let improvement = (last_mid - first_mid).abs();
        if improvement < self.improvement_epsilon {
            return Some(CritiqueResult::StagnantBelief {
                stall_count: self.stall_patience,
            });
        }

        None
    }

    /// Checks the most recent step for a weak premise.
    fn detect_weak_premise<'a>(&self, steps: &[StepView<'a>]) -> Option<CritiqueResult> {
        let last = steps.last()?;

        let word_count = last.premise.split_whitespace().count();
        if word_count < 5 {
            return Some(CritiqueResult::WeakPremise {
                step: last.index,
                suggestion: "Gather more context before forming a conclusion on this topic."
                    .to_string(),
            });
        }

        if last.belief.is_unresolved() {
            return Some(CritiqueResult::WeakPremise {
                step: last.index,
                suggestion: format!(
                    "Belief interval is too wide ({:.2}-{:.2}). Try a more specific query.",
                    last.belief.belief, last.belief.plausibility
                ),
            });
        }

        None
    }

    /// Returns `true` when token overlap between `a` and `b` exceeds 0.5.
    fn high_token_overlap(a: &str, b: &str) -> bool {
        let tokens_a: HashSet<String> = a
            .split_whitespace()
            .map(|w| w.to_ascii_lowercase())
            .filter(|w| w.len() > 2)
            .collect();
        let tokens_b: HashSet<String> = b
            .split_whitespace()
            .map(|w| w.to_ascii_lowercase())
            .filter(|w| w.len() > 2)
            .collect();

        if tokens_a.is_empty() || tokens_b.is_empty() {
            return false;
        }

        let overlap = tokens_a.intersection(&tokens_b).count();
        let min_len = tokens_a.len().min(tokens_b.len()) as f64;
        overlap as f64 / min_len > 0.5
    }

    /// Returns `true` when `text` contains any of the negation markers.
    fn contains_negation(text: &str, markers: &[&str]) -> bool {
        let lower = text.to_ascii_lowercase();
        markers.iter().any(|m| {
            lower
                .split_whitespace()
                .any(|w| w.trim_matches(|c: char| !c.is_ascii_alphabetic()) == *m)
        })
    }
}

impl Default for InternalCritic {
    fn default() -> Self {
        Self::new(0.02, 3)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
