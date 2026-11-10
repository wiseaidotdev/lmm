// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Dempster-Shafer Belief Intervals
//!
//! This module implements the **Dempster-Shafer theory of evidence** as applied
//! to reasoning uncertainty in the [`crate::cognition::thought`] pipeline.
//!
//! Rather than tracking a single probability estimate, each piece of evidence
//! is represented as a `[belief, plausibility]` interval:
//!
//! - **Belief** (`Bel`): the minimum degree of support, the mass of evidence that
//!   *directly* supports the proposition.
//! - **Plausibility** (`Pl`): the maximum possible belief, `1 - Bel(¬proposition)`.
//!
//! The interval `[Bel, Pl]` represents epistemic uncertainty. A narrow interval
//! indicates high confidence; a wide interval signals that more evidence is needed.
//!
//! ## Combination
//!
//! Two independent belief intervals are fused via **Dempster's orthogonal sum**:
//!
//! ```text
//! bel_combined = (bel_A + bel_B − bel_A × bel_B) / K
//! pl_combined  = min(pl_A, pl_B) / K            (conservative fusion)
//! K            = 1 − conflict                   (normalisation constant)
//! conflict     = max(0, bel_A + bel_B − 1)
//! ```
//!
//! ## See Also
//!
//! * [Dempster-Shafer theory - Wikipedia](https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory)
//! * [Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton UP.](https://en.wikipedia.org/wiki/Glenn_Shafer)

/// A belief interval `[belief, plausibility]` representing uncertain evidence.
///
/// Both values are guaranteed to remain in `[0.0, 1.0]` with `belief ≤ plausibility`.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::belief::BeliefInterval;
///
/// let ignorant = BeliefInterval::ignorant();
/// assert_eq!(ignorant.belief, 0.0);
/// assert_eq!(ignorant.plausibility, 1.0);
///
/// let certain = BeliefInterval::certain();
/// assert_eq!(certain.belief, 1.0);
/// assert_eq!(certain.plausibility, 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefInterval {
    /// Lower bound: minimum degree of supported belief, ∈ [0, 1].
    pub belief: f64,
    /// Upper bound: maximum possible belief, ∈ [0, 1], ≥ belief.
    pub plausibility: f64,
}

impl BeliefInterval {
    /// Constructs a `BeliefInterval`, clamping and normalising inputs.
    ///
    /// Ensures `belief ≤ plausibility` and both values are in `[0, 1]`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// let b = BeliefInterval::new(0.6, 0.9);
    /// assert_eq!(b.belief, 0.6);
    /// assert_eq!(b.plausibility, 0.9);
    ///
    /// let swapped = BeliefInterval::new(0.9, 0.3);
    /// assert!(swapped.belief <= swapped.plausibility);
    /// ```
    pub fn new(belief: f64, plausibility: f64) -> Self {
        let b = belief.clamp(0.0, 1.0);
        let p = plausibility.clamp(0.0, 1.0);
        let (lo, hi) = if b <= p { (b, p) } else { (p, b) };
        Self {
            belief: lo,
            plausibility: hi,
        }
    }

    /// Returns the **complete ignorance** interval `[0, 1]`.
    ///
    /// Used as the starting state before any evidence is observed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// let b = BeliefInterval::ignorant();
    /// assert_eq!(b.belief, 0.0);
    /// assert_eq!(b.plausibility, 1.0);
    /// assert_eq!(b.interval_width(), 1.0);
    /// ```
    pub fn ignorant() -> Self {
        Self {
            belief: 0.0,
            plausibility: 1.0,
        }
    }

    /// Returns the **complete certainty** interval `[1, 1]`.
    ///
    /// Used when a proposition is fully established.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// let b = BeliefInterval::certain();
    /// assert!(b.is_actionable(0.9));
    /// ```
    pub fn certain() -> Self {
        Self {
            belief: 1.0,
            plausibility: 1.0,
        }
    }

    /// Derives a belief interval from an **evidence overlap ratio**.
    ///
    /// `found` is the number of goal tokens found in the evidence; `total` is
    /// the total number of goal tokens.  When all tokens are found the result
    /// approaches certainty; when none are found it stays near ignorance.
    ///
    /// # Arguments
    ///
    /// * `found` - Number of goal tokens present in the observation.
    /// * `total` - Total goal tokens (denominator).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// let full = BeliefInterval::from_evidence(5, 5);
    /// assert!(full.belief >= 0.9);
    ///
    /// let none = BeliefInterval::from_evidence(0, 5);
    /// assert_eq!(none.belief, 0.0);
    /// ```
    pub fn from_evidence(found: usize, total: usize) -> Self {
        if total == 0 {
            return Self::ignorant();
        }
        let ratio = found as f64 / total as f64;
        let bel = (ratio * 0.9).clamp(0.0, 1.0);
        let pl = (ratio * 0.9 + 0.1).clamp(0.0, 1.0);
        Self::new(bel, pl)
    }

    /// Fuses two independent belief intervals via **Dempster's orthogonal sum**.
    ///
    /// When the two sources are highly conflicting (conflict > 0.9), the result
    /// falls back to the average to avoid degenerate outcomes from near-total
    /// conflict normalisation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// let a = BeliefInterval::new(0.6, 0.9);
    /// let b = BeliefInterval::new(0.5, 0.8);
    /// let combined = a.combine(&b);
    /// assert!(combined.belief >= a.belief);
    /// assert!(combined.belief <= combined.plausibility);
    /// ```
    pub fn combine(&self, other: &Self) -> Self {
        let conflict = (self.belief + other.belief - 1.0).max(0.0);
        let k = 1.0 - conflict;

        if k < 0.1 {
            let avg_bel = (self.belief + other.belief) / 2.0;
            let avg_pl = (self.plausibility + other.plausibility) / 2.0;
            return Self::new(avg_bel, avg_pl);
        }

        let bel_raw = self.belief + other.belief - self.belief * other.belief;
        let pl_raw = self.plausibility.min(other.plausibility);

        Self::new((bel_raw / k).clamp(0.0, 1.0), (pl_raw / k).clamp(0.0, 1.0))
    }

    /// Returns the **midpoint** of the interval: `(belief + plausibility) / 2`.
    ///
    /// Used as a single-value summary for ranking and comparison.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// let b = BeliefInterval::new(0.4, 0.8);
    /// assert!((b.mid() - 0.6).abs() < 1e-9);
    /// ```
    pub fn mid(&self) -> f64 {
        (self.belief + self.plausibility) / 2.0
    }

    /// Returns the **width** of the interval: `plausibility - belief`.
    ///
    /// A narrow width (< 0.2) indicates high confidence in either direction.
    /// A wide width (> 0.6) indicates significant remaining uncertainty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// assert!((BeliefInterval::ignorant().interval_width() - 1.0).abs() < 1e-9);
    /// assert_eq!(BeliefInterval::certain().interval_width(), 0.0);
    /// ```
    pub fn interval_width(&self) -> f64 {
        self.plausibility - self.belief
    }

    /// Returns `true` when the interval midpoint exceeds `threshold`.
    ///
    /// An actionable belief has enough support to drive a decision, further
    /// evidence collection can be stopped and a conclusion committed to.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// assert!(BeliefInterval::new(0.7, 0.9).is_actionable(0.6));
    /// assert!(!BeliefInterval::new(0.1, 0.4).is_actionable(0.6));
    /// ```
    pub fn is_actionable(&self, threshold: f64) -> bool {
        self.mid() > threshold
    }

    /// Returns `true` when the interval is so wide that the proposition is
    /// effectively unresolved (`interval_width > 0.6` and `belief < 0.3`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::belief::BeliefInterval;
    ///
    /// assert!(BeliefInterval::ignorant().is_unresolved());
    /// assert!(!BeliefInterval::certain().is_unresolved());
    /// ```
    pub fn is_unresolved(&self) -> bool {
        self.interval_width() > 0.6 && self.belief < 0.3
    }
}

impl Default for BeliefInterval {
    fn default() -> Self {
        Self::ignorant()
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
