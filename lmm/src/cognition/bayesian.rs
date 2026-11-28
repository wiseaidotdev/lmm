// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Bayesian Belief Network
//!
//! Builds a small probabilistic graph from KB text chunks using sentence
//! co-occurrence to establish edges, then propagates beliefs via Pearl-style
//! λ/π message passing.
//!
//! ## See Also
//!
//! - [Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.](https://dl.acm.org/doi/book/10.5555/534975)
//! - [Bayesian network - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_network)
//! - [Belief propagation - Wikipedia](https://en.wikipedia.org/wiki/Belief_propagation)
//!
//! ## Quick example
//!
//! ```rust
//! use lmm::cognition::bayesian::{BeliefNet, BeliefPropagator};
//!
//! let chunks = ["water hydrates cells", "hydration supports health"];
//! let mut net = BeliefNet::from_chunks(&chunks, "water");
//! let posterior = BeliefPropagator::converge(&mut net, 20);
//! assert!(posterior >= 0.0 && posterior <= 1.0);
//! ```

pub mod propagation;

pub use propagation::BeliefPropagator;

use crate::constants::tram_loop::BAYESIAN_MAX_PROPAGATION_ROUNDS;
use crate::constants::tram_loop::{
    BAYESIAN_MAX_NODES, BAYESIAN_MAX_PRIOR, BAYESIAN_MIN_EDGE_COOCCURRENCE,
    BAYESIAN_MIN_EVIDENCE_POSTERIOR,
};

/// A single node in the Bayesian belief network.
#[derive(Debug, Clone)]
pub struct BayesNode {
    /// Human-readable token label (single word or short phrase).
    pub label: String,
    /// Prior probability P(node=true) derived from corpus term frequency.
    pub prior: f64,
    /// Likelihood P(evidence | node=true), set during evidence updates.
    pub likelihood: f64,
    /// Current posterior P(node=true | evidence), updated each propagation round.
    pub posterior: f64,
    /// Indices into [`BeliefNet::nodes`] of this node's children.
    pub children: Vec<usize>,
}

impl BayesNode {
    /// Constructs a new node with the given label and prior.
    ///
    /// `likelihood` and `posterior` are initialised to `prior`.
    pub fn new(label: &str, prior: f64) -> Self {
        let clamped = prior.clamp(0.0, BAYESIAN_MAX_PRIOR);
        Self {
            label: label.to_string(),
            prior: clamped,
            likelihood: clamped,
            posterior: clamped,
            children: Vec::new(),
        }
    }
}

/// A small Bayesian belief network built from KB text.
pub struct BeliefNet {
    /// All nodes in the network.
    pub nodes: Vec<BayesNode>,
    /// Topic query the network was built for.
    pub topic: String,
    /// Index of the topic's root node (0 if network is empty).
    pub topic_node_idx: usize,
}

impl BeliefNet {
    /// Builds a belief network from `chunks` by extracting term frequencies
    /// and sentence co-occurrence edges.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Slice of text strings from the knowledge index.
    /// * `topic`  - The subject query; the topic's root node is promoted.
    pub fn from_chunks<S: AsRef<str>>(chunks: &[S], topic: &str) -> Self {
        let joined: String = chunks
            .iter()
            .map(|c| c.as_ref())
            .collect::<Vec<_>>()
            .join(" ");
        let sentences: Vec<&str> = joined.split(['.', '!', '?', '\n']).collect();

        let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let topic_lc = topic.to_ascii_lowercase();
        let total_words: usize = sentences
            .iter()
            .flat_map(|s| {
                s.split_whitespace().filter(|w| w.len() >= 4).map(|w| {
                    w.to_ascii_lowercase()
                        .trim_matches(|c: char| !c.is_alphabetic())
                        .to_string()
                })
            })
            .inspect(|w| {
                *freq.entry(w.clone()).or_default() += 1;
            })
            .count();
        let total_words = total_words.max(1);

        let mut labels: Vec<String> = freq
            .iter()
            .filter(|&(_, &c)| c >= 2)
            .map(|(k, _)| k.clone())
            .collect();
        labels.sort_by(|a, b| freq.get(b).unwrap_or(&0).cmp(freq.get(a).unwrap_or(&0)));
        labels.truncate(BAYESIAN_MAX_NODES.saturating_sub(1));

        if !labels.contains(&topic_lc) {
            labels.insert(0, topic_lc.clone());
        }

        let mut nodes: Vec<BayesNode> = labels
            .iter()
            .map(|label| {
                let count = *freq.get(label).unwrap_or(&1) as f64;
                let prior = (count / total_words as f64).clamp(0.0, BAYESIAN_MAX_PRIOR);
                BayesNode::new(label, prior)
            })
            .collect();

        let label_to_idx: std::collections::HashMap<&str, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.as_str(), i))
            .collect();

        let mut cooccurrence: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();

        for sentence in &sentences {
            let words: Vec<String> = sentence
                .split_whitespace()
                .filter(|w| w.len() >= 4)
                .map(|w| {
                    w.to_ascii_lowercase()
                        .trim_matches(|c: char| !c.is_alphabetic())
                        .to_string()
                })
                .filter(|w| label_to_idx.contains_key(w.as_str()))
                .collect();
            for i in 0..words.len() {
                for j in (i + 1)..words.len().min(i + 6) {
                    if let (Some(&a), Some(&b)) = (
                        label_to_idx.get(words[i].as_str()),
                        label_to_idx.get(words[j].as_str()),
                    ) {
                        let pair = if a < b { (a, b) } else { (b, a) };
                        *cooccurrence.entry(pair).or_default() += 1;
                    }
                }
            }
        }

        for ((parent, child), count) in &cooccurrence {
            if *count >= BAYESIAN_MIN_EDGE_COOCCURRENCE
                && *parent != *child
                && !nodes[*parent].children.contains(child)
            {
                nodes[*parent].children.push(*child);
            }
        }

        let topic_node_idx = label_to_idx.get(topic_lc.as_str()).copied().unwrap_or(0);

        Self {
            nodes,
            topic: topic.to_string(),
            topic_node_idx,
        }
    }

    /// Updates node likelihoods from `evidence` text and returns the network.
    ///
    /// For each evidence token matching a node label, the node's likelihood is
    /// raised toward 1.0 proportional to its occurrence density in the evidence.
    pub fn update_with_evidence(&mut self, evidence: &str) {
        let tokens: Vec<String> = evidence
            .split_whitespace()
            .map(|w| {
                w.to_ascii_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect();
        let total = tokens.len().max(1) as f64;
        for node in &mut self.nodes {
            let hits = tokens
                .iter()
                .filter(|t| t.as_str() == node.label.as_str())
                .count() as f64;
            if hits > 0.0 {
                let boost = (hits / total).clamp(0.0, 1.0 - node.likelihood);
                node.likelihood = (node.likelihood + boost).clamp(0.0, 1.0);
            }
        }
    }

    /// Runs propagation over the network given prompt evidence tokens, returning
    /// the final topic posterior.
    ///
    /// # Arguments
    ///
    /// * `evidence_tokens` - Words from the prompt, used to seed likelihoods.
    pub fn propagate(&mut self, evidence_tokens: &[&str]) -> f64 {
        let evidence = evidence_tokens.join(" ");
        self.update_with_evidence(&evidence);
        propagation::BeliefPropagator::converge(self, BAYESIAN_MAX_PROPAGATION_ROUNDS)
    }

    /// Returns the posterior of the topic root node.
    pub fn topic_posterior(&self) -> f64 {
        self.nodes
            .get(self.topic_node_idx)
            .map(|n| n.posterior)
            .unwrap_or(BAYESIAN_MIN_EVIDENCE_POSTERIOR)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
