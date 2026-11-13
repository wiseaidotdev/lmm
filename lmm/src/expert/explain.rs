// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Expert System: Explanation Facility
//!
//! Implements the **how/why explanation facility**.
//!
//! When an expert system derives a conclusion, it must be able to explain:
//!
//! - **"How"**: which rules and sub-goals produced the conclusion.
//! - **"Why"**: why a particular piece of information was needed during reasoning.
//!
//! The explanation is stored as an AND/OR proof tree and can be rendered either
//! as a human-readable trace or as a compact prose paragraph suitable for
//! inclusion in TRAM's output.

use crate::constants::expert::DEFAULT_RULE_LABEL;
use crate::expert::uncertainty::CertaintyFactor;

/// A single step in the derivation chain.
///
/// ```text
/// Goal was derived by Rule from Sub-goals
/// ```
///
/// For ground facts (axioms), `derived_by` is `None` and `from_conditions`
/// is empty.
///
/// # Examples
///
/// ```rust
/// use lmm::expert::explain::ProofStep;
///
/// let step = ProofStep {
///     conclusion: "rust is memory-safe".to_string(),
///     derived_by: Some("rule_ownership".to_string()),
///     from_conditions: vec![
///         "rust has ownership".to_string(),
///         "rust has borrow-checker".to_string(),
///     ],
///     certainty_factor: 0.81,
/// };
/// assert!(step.derived_by.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// The conclusion established at this step.
    pub conclusion: String,
    /// The rule that derived this conclusion, or `None` if it is a ground fact.
    pub derived_by: Option<String>,
    /// The sub-conditions that were satisfied to fire the rule.
    pub from_conditions: Vec<String>,
    /// The certainty factor at this step.
    pub certainty_factor: f64,
}

impl ProofStep {
    /// Creates a **ground-fact** step (the leaf of a proof tree).
    ///
    /// # Arguments
    ///
    /// * `fact` - The ground assertion that was directly observed or retrieved.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn told(fact: impl Into<String>) -> Self {
        Self {
            conclusion: fact.into(),
            derived_by: None,
            from_conditions: Vec::new(),
            certainty_factor: 1.0,
        }
    }

    /// Creates a **derived** step from a rule application.
    ///
    /// # Arguments
    ///
    /// * `conclusion`  - The derived assertion.
    /// * `rule_id`     - The rule that was applied.
    /// * `conditions`  - The sub-goals that were satisfied.
    /// * `cf`          - The certainty factor of this derivation.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn derived(
        conclusion: impl Into<String>,
        rule_id: impl Into<String>,
        conditions: Vec<String>,
        cf: f64,
    ) -> Self {
        Self {
            conclusion: conclusion.into(),
            derived_by: Some(rule_id.into()),
            from_conditions: conditions,
            certainty_factor: cf,
        }
    }

    /// Renders this step as a single human-readable line.
    ///
    /// # Time Complexity
    ///
    /// O(|conclusion| + |from_conditions|)
    pub fn to_line(&self) -> String {
        if let Some(rule) = &self.derived_by {
            format!(
                "{} (derived by {}, CF={:.2})",
                self.conclusion, rule, self.certainty_factor
            )
        } else {
            format!("{} (observed)", self.conclusion)
        }
    }
}

/// A flattened explanation trace for the entire proof.
///
/// Stores all proof steps in derivation order: deepest sub-goals first,
/// top-level conclusion last.
///
/// # Examples
///
/// ```rust
/// use lmm::expert::explain::{ExplanationTrace, ProofStep};
///
/// let mut trace = ExplanationTrace::new();
/// trace.push(ProofStep::told("rust has ownership"));
/// trace.push(ProofStep::derived("rust is memory-safe", "rule_ownership", vec!["rust has ownership".to_string()], 0.9));
/// let prose = trace.to_prose();
/// assert!(prose.contains("memory-safe"));
/// ```
#[derive(Debug, Default, Clone)]
pub struct ExplanationTrace {
    /// All proof steps in derivation order (leaves first, root last).
    pub steps: Vec<ProofStep>,
}

impl ExplanationTrace {
    /// Creates an empty trace.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a proof step to the trace.
    ///
    /// # Time Complexity
    ///
    /// O(1) amortised
    pub fn push(&mut self, step: ProofStep) {
        self.steps.push(step);
    }

    /// Returns the top-level conclusion sentence (the last step's conclusion).
    ///
    /// Returns an empty string if the trace is empty.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn conclusion_sentence(&self) -> String {
        self.steps
            .last()
            .map(|s| s.conclusion.clone())
            .unwrap_or_default()
    }

    /// Renders the full trace as compact prose suitable for TRAM output.
    ///
    /// Example output:
    ///
    /// ```text
    /// Based on observed facts (rust has ownership, rust has borrow-checker),
    /// rule_ownership establishes that rust is memory-safe with high confidence (CF=0.81).
    /// ```
    ///
    /// # Time Complexity
    ///
    /// O(|steps|)
    pub fn to_prose(&self) -> String {
        if self.steps.is_empty() {
            return String::new();
        }

        let top = self.steps.last().unwrap();
        let facts: Vec<String> = self
            .steps
            .iter()
            .filter(|s| s.derived_by.is_none())
            .map(|s| s.conclusion.clone())
            .collect();

        let derived: Vec<&ProofStep> = self
            .steps
            .iter()
            .filter(|s| s.derived_by.is_some())
            .collect();

        let mut parts: Vec<String> = Vec::new();

        if !facts.is_empty() {
            let fact_list = if facts.len() == 1 {
                facts[0].clone()
            } else {
                let (last, rest) = facts.split_last().unwrap();
                format!("{}, and {}", rest.join(", "), last)
            };
            parts.push(format!("Based on observed knowledge ({fact_list})"));
        }

        for step in &derived {
            let rule = step.derived_by.as_deref().unwrap_or(DEFAULT_RULE_LABEL);
            let label = CertaintyFactor::new(step.certainty_factor).label();
            parts.push(format!(
                "{rule} establishes that {} with {label} (CF={:.2})",
                step.conclusion, step.certainty_factor
            ));
        }

        if derived.is_empty() {
            let label = CertaintyFactor::new(top.certainty_factor).label();
            parts.push(format!(
                "The conclusion is: {} ({label}, CF={:.2})",
                top.conclusion, top.certainty_factor
            ));
        }

        parts.join(". ") + "."
    }

    /// Returns the number of steps in the trace.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` if the trace has no steps.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
