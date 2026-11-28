// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Bayesian Belief Propagation
//!
//! Implements Pearl-style λ/π message passing over a [`BeliefNet`].
//!
//! Each propagation round:
//!
//! 1. **π messages** flow from parent → child, expressing how much the parent
//!    "supports" the child being true given the current evidence state.
//! 2. **λ messages** flow from child → parent, expressing how much the child's
//!    evidence is explained *away* by the parent's state.
//! 3. Node posteriors are updated as the product of π and λ values, normalised.
//!
//! Rounds continue until the maximum node-posterior change drops below
//! [`BAYESIAN_CONVERGENCE_EPSILON`] or [`BAYESIAN_MAX_PROPAGATION_ROUNDS`] is
//! reached.
//!
//! ## See Also
//!
//! - [Belief propagation - Wikipedia](https://en.wikipedia.org/wiki/Belief_propagation)
//! - [Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.](https://dl.acm.org/doi/book/10.5555/534975)
//! - [Sum-product algorithm - Wikipedia](https://en.wikipedia.org/wiki/Belief_propagation#Factor_graph_and_the_sum-product_algorithm)

use crate::cognition::bayesian::BeliefNet;
use crate::constants::tram_loop::{BAYESIAN_CONVERGENCE_EPSILON, BAYESIAN_MAX_PROPAGATION_ROUNDS};

/// Stateless struct grouping Pearl λ/π propagation operations over a [`BeliefNet`].
pub struct BeliefPropagator;

impl BeliefPropagator {
    /// Runs one full λ/π propagation round over `net`.
    ///
    /// Returns the maximum absolute change in any node's posterior after the round.
    ///
    /// # Arguments
    ///
    /// * `net` - Mutable reference to the belief network to propagate over.
    pub fn propagate_round(net: &mut BeliefNet) -> f64 {
        let n = net.nodes.len();
        if n == 0 {
            return 0.0;
        }

        let old_posteriors: Vec<f64> = net.nodes.iter().map(|node| node.posterior).collect();

        let pi_vals: Vec<f64> = (0..n)
            .map(|i| {
                let parents: Vec<usize> = net
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.children.contains(&i))
                    .map(|(idx, _)| idx)
                    .collect();
                if parents.is_empty() {
                    net.nodes[i].prior
                } else {
                    parents
                        .iter()
                        .map(|&p| net.nodes[p].posterior * net.nodes[i].likelihood)
                        .sum::<f64>()
                        .min(1.0)
                }
            })
            .collect();

        let lambda_vals: Vec<f64> = (0..n)
            .map(|i| {
                let children = net.nodes[i].children.clone();
                if children.is_empty() {
                    net.nodes[i].likelihood
                } else {
                    children
                        .iter()
                        .map(|&c| net.nodes[c].likelihood * old_posteriors[c])
                        .sum::<f64>()
                        .clamp(1e-9, 1.0)
                }
            })
            .collect();

        let mut max_delta: f64 = 0.0;
        for i in 0..n {
            let raw = pi_vals[i] * lambda_vals[i];
            let new_posterior = raw.clamp(0.0, 1.0);
            let delta = (new_posterior - old_posteriors[i]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            net.nodes[i].posterior = new_posterior;
        }
        max_delta
    }

    /// Iterates propagation rounds until convergence or the round cap is reached.
    ///
    /// Returns the topic-node posterior after convergence.
    ///
    /// # Arguments
    ///
    /// * `net`        - Mutable reference to the belief network.
    /// * `max_rounds` - Upper bound on rounds (clamped to [`BAYESIAN_MAX_PROPAGATION_ROUNDS`]).
    pub fn converge(net: &mut BeliefNet, max_rounds: usize) -> f64 {
        let cap = max_rounds.min(BAYESIAN_MAX_PROPAGATION_ROUNDS);
        for _ in 0..cap {
            let delta = Self::propagate_round(net);
            if delta < BAYESIAN_CONVERGENCE_EPSILON {
                break;
            }
        }
        net.topic_posterior()
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
