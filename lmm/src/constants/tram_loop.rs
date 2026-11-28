// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Loop Orchestrator Constants

/// Default minimum quality-delta required between consecutive loop passes to
/// continue iterating.
pub const LOOP_MIN_QUALITY_DELTA: f64 = 0.05;

/// Safety cap on the maximum number of loop passes, regardless of `--loop`.
pub const LOOP_MAX_COUNT: usize = 10;

/// Section divider label for each loop pass in verbose output.
pub const LOOP_PASS_DIVIDER_LABEL: &str = "Loop Pass";

/// Emoji prefix for loop-pass start log lines.
pub const LOOP_PASS_EMOJI: &str = "🔁";

/// Format string for the per-pass quality log line.
pub const LOOP_PASS_QUALITY_FMT: &str =
    "  🔁 Pass {}/{}: quality={:.3} delta={:+.3} belief_mid={:.3}";

/// Format string for the Bayesian belief-update log line.
pub const BAYESIAN_UPDATE_FMT: &str = "  🧠 Bayesian: prior={:.3} posterior={:.3} delta={:+.3}";

/// Log line emitted when the loop exits early due to quality convergence.
pub const LOOP_CONVERGED_FMT: &str =
    "  ✅ Loop converged at pass {} (delta={:.3} < threshold={:.3})";

/// Log line emitted when the loop reaches its iteration cap.
pub const LOOP_EXHAUSTED_FMT: &str = "  🔚 Loop exhausted {} passes (final quality={:.3})";

/// Minimum Bayesian posterior to treat the network as having produced
/// meaningful evidence (below this the update is treated as noise).
pub const BAYESIAN_MIN_EVIDENCE_POSTERIOR: f64 = 0.05;

/// Maximum number of propagation rounds in the Bayesian belief network
/// before halting (prevents infinite loops in cyclic graphs).
pub const BAYESIAN_MAX_PROPAGATION_ROUNDS: usize = 20;

/// Convergence epsilon for Bayesian propagation: rounds stop when the
/// maximum node-posterior change across the network falls below this.
pub const BAYESIAN_CONVERGENCE_EPSILON: f64 = 1e-4;

/// Weight of Bayesian posterior when fusing with DS belief interval midpoint.
///
/// `fused = bayes_weight * posterior + (1 - bayes_weight) * ds_mid`.
pub const BAYESIAN_FUSION_WEIGHT: f64 = 0.4;

/// Minimum co-occurrence count for a Bayesian network edge to be established.
pub const BAYESIAN_MIN_EDGE_COOCCURRENCE: usize = 2;

/// Maximum number of nodes in the Bayesian belief network (memory cap).
pub const BAYESIAN_MAX_NODES: usize = 128;

/// Default prior probability assigned to a node when no KB evidence is found.
pub const BAYESIAN_DEFAULT_PRIOR: f64 = 0.1;

/// Maximum prior probability clamp, avoids fully-certain priors from raw
/// corpus frequency that would make updates non-informative.
pub const BAYESIAN_MAX_PRIOR: f64 = 0.85;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
