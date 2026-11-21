// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Causal reasoning methods for [`LmmAgent`].
//!
//! Covers [`attribute_causes`](LmmAgent::attribute_causes) and
//! [`form_hypotheses`](LmmAgent::form_hypotheses).

use lmm::causal::CausalGraph;
use lmm::cognition::attribution::{AttributionReport, CausalAttributor};
use lmm::cognition::hypothesis::{Hypothesis, HypothesisGenerator};
use std::collections::HashMap;

use super::LmmAgent;

impl LmmAgent {
    /// Attributes the outcome of `outcome_var` in `graph` to its causal parents
    /// by running Pearl *do*-calculus counterfactuals on each parent.
    ///
    /// Returns an [`AttributionReport`] with normalised weights sorted
    /// highest-first, or `None` when `outcome_var` has no parents.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::causal::CausalGraph;
    /// use lmm_agent::agent::LmmAgent;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("cause", Some(2.0));
    /// g.add_node("effect", None);
    /// g.add_edge("cause", "effect", Some(1.0)).unwrap();
    /// g.forward_pass().unwrap();
    ///
    /// let agent = LmmAgent::new("Analyst".into(), "Causal analysis.".into());
    /// let report = agent.attribute_causes(&g, "effect").unwrap();
    /// assert_eq!(report.weights[0].0, "cause");
    /// ```
    pub fn attribute_causes(
        &self,
        graph: &CausalGraph,
        outcome_var: &str,
    ) -> anyhow::Result<AttributionReport> {
        CausalAttributor::attribute(graph, outcome_var).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Generates causal hypotheses for variables whose observed values are not
    /// explained by the current `graph` structure.
    ///
    /// Returns up to `max_hypotheses` candidate new edges ranked by
    /// explanatory power, highest first.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::causal::CausalGraph;
    /// use lmm_agent::agent::LmmAgent;
    /// use std::collections::HashMap;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("x", Some(1.0));
    /// g.add_node("y", Some(0.0));
    ///
    /// let mut observed = HashMap::new();
    /// observed.insert("y".to_string(), 0.9);
    ///
    /// let agent = LmmAgent::new("Scientist".into(), "Discover causal laws.".into());
    /// let hypotheses = agent.form_hypotheses(&g, &observed, 5).unwrap();
    /// assert!(!hypotheses.is_empty());
    /// ```
    pub fn form_hypotheses(
        &self,
        graph: &CausalGraph,
        observed: &HashMap<String, f64>,
        max_hypotheses: usize,
    ) -> anyhow::Result<Vec<Hypothesis>> {
        let r#gen = HypothesisGenerator::new(0.05, max_hypotheses);
        r#gen
            .generate(graph, observed)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
