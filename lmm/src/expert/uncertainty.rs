// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Expert System: Certainty Factor Model
//!
//! Implements the **Prospector certainty model**.
//!
//! In the Prospector model (originally applied to mineral exploration and fault
//! diagnosis), uncertainty is expressed as a *subjective certainty* ∈ [0, 1]
//! rather than a strict probability.  The model uses two combination rules:
//!
//! | Operation | Formula | Semantics |
//! |---|---|---|
//! | Sequential combination | `cf₁ + cf₂ × (1 - cf₁)` | Two independent pieces of evidence accumulate |
//! | Rule propagation | `evidence_cf × rule_cf` | Uncertain evidence weakens the rule conclusion |
//!
//! These operations are intentionally simpler than Bayes' theorem. human experts do not think in precise
//! probabilities; a subjective certainty scale is a better fit for expert knowledge.

/// A Prospector-style certainty factor in the interval [0.0, 1.0].
///
/// - `0.0`: no evidence; the assertion is completely unknown.
/// - `1.0`: absolute certainty; the assertion is known to be true.
///
/// Values are always clamped to [0.0, 1.0] on construction.
///
/// # Examples
///
/// ```rust
/// use lmm::expert::uncertainty::CertaintyFactor;
///
/// let cf_a = CertaintyFactor::new(0.6);
/// let cf_b = CertaintyFactor::new(0.5);
/// let combined = cf_a.combine(&cf_b);
/// assert!(combined.value() > 0.6); // accumulates evidence
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CertaintyFactor(f64);

impl CertaintyFactor {
    /// Creates a new `CertaintyFactor` clamped to [0.0, 1.0].
    ///
    /// # Arguments
    ///
    /// * `value` - Raw certainty value; clamped if outside [0.0, 1.0].
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Returns the raw `f64` certainty value.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Combines two independent pieces of evidence.
    ///
    /// Uses the Prospector sequential combination formula:
    ///
    /// ```text
    /// cf_combined = cf₁ + cf₂ × (1 - cf₁)
    /// ```
    ///
    /// This is monotonically increasing: adding evidence never reduces certainty.
    ///
    /// # Arguments
    ///
    /// * `other` - A second, independent certainty factor.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn combine(&self, other: &CertaintyFactor) -> CertaintyFactor {
        let combined = self.0 + other.0 * (1.0 - self.0);
        CertaintyFactor::new(combined)
    }

    /// Propagates this certainty through a rule with the given rule certainty factor.
    ///
    /// Models the weakening of a conclusion when the evidence itself is uncertain.
    ///
    /// ```text
    /// cf_propagated = evidence_cf × rule_cf
    /// ```
    ///
    /// # Arguments
    ///
    /// * `rule_cf` - The intrinsic certainty factor of the rule (∈ [0.0, 1.0]).
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn propagate(&self, rule_cf: f64) -> CertaintyFactor {
        CertaintyFactor::new(self.0 * rule_cf.clamp(0.0, 1.0))
    }

    /// Returns `true` when this certainty factor meets or exceeds `threshold`.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum acceptable certainty (e.g. 0.4 for expert inference).
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn is_sufficient(&self, threshold: f64) -> bool {
        self.0 >= threshold
    }

    /// Returns a human-readable label for the certainty level.
    ///
    /// | Range | Label |
    /// |---|---|
    /// | [0.0, 0.3)  | "low confidence"     |
    /// | [0.3, 0.6)  | "moderate confidence" |
    /// | [0.6, 0.85) | "high confidence"    |
    /// | [0.85, 1.0] | "very high confidence" |
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn label(&self) -> &'static str {
        if self.0 < 0.3 {
            "low confidence"
        } else if self.0 < 0.6 {
            "moderate confidence"
        } else if self.0 < 0.85 {
            "high confidence"
        } else {
            "very high confidence"
        }
    }
}

/// Combines a slice of certainty factors using the Prospector sequential formula.
///
/// Equivalent to folding `CertaintyFactor::combine` over the entire slice.
/// Returns `CertaintyFactor::new(0.0)` for an empty slice.
///
/// # Arguments
///
/// * `factors` - Slice of certainty factors to combine.
///
/// # Time Complexity
///
/// O(|factors|)
pub fn combine_all(factors: &[CertaintyFactor]) -> CertaintyFactor {
    factors
        .iter()
        .fold(CertaintyFactor::new(0.0), |acc, cf| acc.combine(cf))
}

/// Derives a certainty factor from a normalised IDF score.
///
/// IDF scores are typically > 1.0.  This function maps them onto [0.1, 0.95]
/// using a logarithmic compression so that common, low-IDF terms do not receive
/// zero certainty and high-IDF rare terms do not saturate to 1.0.
///
/// ```text
/// cf = ln(1 + idf) / ln(1 + max_idf) × 0.85 + 0.1
/// ```
///
/// # Arguments
///
/// * `idf`     - Raw IDF weight for a single term.
/// * `max_idf` - Maximum IDF weight observed in the corpus (for normalisation).
///
/// # Time Complexity
///
/// O(1)
pub fn cf_from_idf(idf: f64, max_idf: f64) -> CertaintyFactor {
    if max_idf <= 0.0 {
        return CertaintyFactor::new(0.1);
    }
    let normalised = (1.0 + idf).ln() / (1.0 + max_idf).ln();
    CertaintyFactor::new(normalised * 0.85 + 0.1)
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
