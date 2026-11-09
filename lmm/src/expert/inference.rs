// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Expert System: Inference Engine
//!
//! Implements **backward-chaining inference**.
//!
//! The inference engine takes a natural-language *goal* (the user's prompt),
//! searches the [`RuleBase`] for rules whose conclusions overlap with the goal,
//! recursively resolves their conditions, and combines evidence using the
//! Prospector certainty model from [`crate::expert::uncertainty`].
//!
//! ## Algorithm
//!
//! ```text
//! reason(Goal):
//!   find rules R whose conclusion matches Goal
//!   for each R:
//!     for each condition C in R.conditions:
//!       if fact_base contains C → cf_condition = 1.0
//!       else → cf_condition = reason(C.to_text()).certainty (recursive)
//!     cf_rule = propagate(cf_conditions_combined, R.certainty_factor)
//!     accumulate cf_rule into result via combine
//!   return best InferenceResult
//! ```
//!
//! Cycles are prevented by the `visited` depth limit (`max_depth`).

use crate::expert::explain::{ExplanationTrace, ProofStep};
use crate::expert::rule::{Rule, RuleBase};
use crate::expert::uncertainty::{CertaintyFactor, combine_all};

/// The result of a single inference run.
///
/// Contains the top-level conclusion, its combined certainty, the full
/// derivation trace, and the depth of the proof tree.
///
/// # Examples
///
/// ```rust
/// use lmm::expert::inference::InferenceResult;
/// use lmm::expert::explain::ExplanationTrace;
///
/// let r = InferenceResult {
///     conclusion: "rust is memory-safe".to_string(),
///     certainty: 0.73,
///     trace: ExplanationTrace::new(),
///     depth: 2,
/// };
/// assert_eq!(r.conclusion, "rust is memory-safe");
/// ```
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The top-level conclusion that was established.
    pub conclusion: String,
    /// Combined Prospector certainty factor ∈ [0.0, 1.0].
    pub certainty: f64,
    /// Full proof trace (how/why explanation).
    pub trace: ExplanationTrace,
    /// Depth of the AND/OR proof tree.
    pub depth: usize,
}

/// Backward-chaining inference engine over a [`RuleBase`].
///
/// Instantiated once per TRAM request; not shared across threads.
///
/// # Examples
///
/// ```rust
/// use lmm::expert::inference::InferenceEngine;
/// use lmm::expert::rule::RuleBase;
///
/// let chunks = [
///     "Rust provides compile-time memory safety guarantees.",
///     "Rust enables safe concurrency without data races.",
/// ];
/// let base = RuleBase::from_chunks(&chunks);
/// let engine = InferenceEngine::new(3, 0.20);
///
/// let _result = engine.reason("rust memory safety", &base);
/// assert!(engine.max_depth == 3);
/// ```
#[derive(Debug, Clone)]
pub struct InferenceEngine {
    /// Maximum backward-chaining depth (prevents infinite recursion on cyclic rules).
    pub max_depth: usize,
    /// Minimum certainty factor required to accept an inference result.
    pub cf_threshold: f64,
}

impl InferenceEngine {
    /// Creates a new `InferenceEngine`.
    ///
    /// # Arguments
    ///
    /// * `max_depth`    - Depth limit for backward chaining (3-5 is typical).
    /// * `cf_threshold` - Minimum certainty to accept (e.g. 0.3).
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn new(max_depth: usize, cf_threshold: f64) -> Self {
        Self {
            max_depth,
            cf_threshold: cf_threshold.clamp(0.0, 1.0),
        }
    }

    /// Performs backward-chaining inference from `goal` using `base`.
    ///
    /// Returns the best-certainty `InferenceResult` if any chain achieves
    /// certainty ≥ `self.cf_threshold`, or `None` if the goal cannot be
    /// established with sufficient confidence.
    ///
    /// # Arguments
    ///
    /// * `goal` - Natural-language goal string (e.g. "rust is fast").
    /// * `base` - The rule base derived from the knowledge index.
    ///
    /// # Time Complexity
    ///
    /// O(|rules|^max_depth) in the worst case; in practice O(|rules| × max_depth)
    /// because most goals match only a small fraction of rules.
    ///
    /// # Space Complexity
    ///
    /// O(max_depth) for the recursion stack.
    pub fn reason(&self, goal: &str, base: &RuleBase) -> Option<InferenceResult> {
        self.reason_inner(goal, base, 0)
    }

    /// Inner recursive helper for backward chaining.
    ///
    /// # Arguments
    ///
    /// * `goal`  - Current sub-goal.
    /// * `base`  - Rule base.
    /// * `depth` - Current recursion depth (stops at `self.max_depth`).
    ///
    /// # Time Complexity
    ///
    /// O(|rules| × max_depth)
    fn reason_inner(&self, goal: &str, base: &RuleBase, depth: usize) -> Option<InferenceResult> {
        if depth >= self.max_depth {
            return None;
        }

        let matching = base.matching_rules(goal);
        if matching.is_empty() {
            return None;
        }

        let mut best: Option<InferenceResult> = None;

        for rule in matching {
            if let Some(result) = self.apply_rule(rule, base, depth) {
                let better = best
                    .as_ref()
                    .map(|b| result.certainty > b.certainty)
                    .unwrap_or(true);
                if better {
                    best = Some(result);
                }
            }
        }

        best.filter(|r| r.certainty >= self.cf_threshold)
    }

    /// Attempts to fire a single rule and produce an `InferenceResult`.
    ///
    /// All conditions are resolved; their certainty factors are combined and
    /// propagated through the rule's own CF.
    ///
    /// # Arguments
    ///
    /// * `rule`  - The rule to apply.
    /// * `base`  - Rule base for condition resolution.
    /// * `depth` - Current recursion depth.
    ///
    /// # Time Complexity
    ///
    /// O(|conditions| × |rules| × max_depth)
    fn apply_rule(&self, rule: &Rule, base: &RuleBase, depth: usize) -> Option<InferenceResult> {
        let mut condition_cfs: Vec<CertaintyFactor> = Vec::new();
        let mut trace = ExplanationTrace::new();
        let mut total_depth = depth;

        for cond in &rule.conditions {
            if base.condition_holds(cond) {
                condition_cfs.push(CertaintyFactor::new(1.0));
                trace.push(ProofStep::told(cond.to_text()));
            } else {
                let sub_goal = cond.to_text();
                match self.reason_inner(&sub_goal, base, depth + 1) {
                    Some(sub_result) => {
                        total_depth = total_depth.max(sub_result.depth);
                        condition_cfs.push(CertaintyFactor::new(sub_result.certainty));
                        for step in sub_result.trace.steps {
                            trace.push(step);
                        }
                    }
                    None => {
                        condition_cfs.push(CertaintyFactor::new(0.0));
                    }
                }
            }
        }

        let combined_evidence = combine_all(&condition_cfs);
        let final_cf = combined_evidence.propagate(rule.certainty_factor);

        if final_cf.value() < self.cf_threshold {
            return None;
        }

        let condition_texts: Vec<String> = rule.conditions.iter().map(|c| c.to_text()).collect();

        trace.push(ProofStep::derived(
            rule.conclusion.clone(),
            rule.id.clone(),
            condition_texts,
            final_cf.value(),
        ));

        Some(InferenceResult {
            conclusion: rule.conclusion.clone(),
            certainty: final_cf.value(),
            trace,
            depth: total_depth + 1,
        })
    }
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
