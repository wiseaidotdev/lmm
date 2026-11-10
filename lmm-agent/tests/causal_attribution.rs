// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::causal::CausalGraph;
use lmm::cognition::attribution::CausalAttributor;
use lmm::cognition::hypothesis::HypothesisGenerator;
use std::collections::HashMap;

fn simple_chain() -> CausalGraph {
    let mut g = CausalGraph::new();
    g.add_node("co2", Some(400.0));
    g.add_node("temp", None);
    g.add_edge("co2", "temp", Some(0.01)).unwrap();
    g
}

#[test]
fn causal_graph_new_is_empty() {
    let g = CausalGraph::new();
    assert_eq!(g.nodes.len(), 0);
    assert_eq!(g.edges.len(), 0);
}

#[test]
fn add_node_increases_node_count() {
    let mut g = CausalGraph::new();
    g.add_node("x", Some(1.0));
    assert_eq!(g.nodes.len(), 1);
}

#[test]
fn add_edge_increases_edge_count() {
    let mut g = CausalGraph::new();
    g.add_node("a", Some(1.0));
    g.add_node("b", None);
    g.add_edge("a", "b", Some(0.5)).unwrap();
    assert_eq!(g.edges.len(), 1);
}

#[test]
fn forward_pass_propagates_value_to_child() {
    let mut g = simple_chain();
    g.forward_pass().unwrap();
    let temp = g.nodes.iter().find(|n| n.name == "temp").unwrap();
    let expected = 400.0 * 0.01;
    assert!(
        (temp.value.unwrap() - expected).abs() < 1e-6,
        "expected temp={expected}, got {:?}",
        temp.value
    );
}

#[test]
fn forward_pass_root_node_value_unchanged() {
    let mut g = simple_chain();
    g.forward_pass().unwrap();
    let co2 = g.nodes.iter().find(|n| n.name == "co2").unwrap();
    assert!((co2.value.unwrap() - 400.0).abs() < 1e-9);
}

#[test]
fn add_edge_to_missing_node_returns_error() {
    let mut g = CausalGraph::new();
    g.add_node("a", Some(1.0));
    let result = g.add_edge("a", "missing", Some(1.0));
    assert!(result.is_err(), "should error when target node missing");
}

#[test]
fn causal_graph_forward_pass_is_idempotent() {
    let mut g = simple_chain();
    g.forward_pass().unwrap();
    let first = g.nodes.iter().find(|n| n.name == "temp").unwrap().value;
    g.forward_pass().unwrap();
    let second = g.nodes.iter().find(|n| n.name == "temp").unwrap().value;
    assert_eq!(first, second, "forward_pass should be idempotent");
}

#[test]
fn parents_returns_correct_parents() {
    let mut g = CausalGraph::new();
    g.add_node("a", Some(1.0));
    g.add_node("b", None);
    g.add_node("c", None);
    g.add_edge("a", "c", Some(1.0)).unwrap();
    g.add_edge("b", "c", Some(0.5)).unwrap();
    let parents = g.parents("c");
    assert!(parents.contains(&"a".to_string()));
    assert!(parents.contains(&"b".to_string()));
}

#[test]
fn causal_attributor_runs_on_simple_graph() {
    let mut g = simple_chain();
    g.forward_pass().unwrap();
    let result = CausalAttributor::attribute(&g, "temp");
    assert!(result.is_ok(), "attribution failed: {:?}", result.err());
}

#[test]
fn attribution_report_weight_for_known_cause() {
    let mut g = simple_chain();
    g.forward_pass().unwrap();
    let report = CausalAttributor::attribute(&g, "temp").unwrap();
    let weight = report.weight_for("co2");
    assert!(weight.is_some(), "co2 should appear in attribution report");
    assert!(weight.unwrap() > 0.0);
}

#[test]
fn attribution_report_weight_for_unknown_returns_none() {
    let mut g = simple_chain();
    g.forward_pass().unwrap();
    let report = CausalAttributor::attribute(&g, "temp").unwrap();
    assert!(report.weight_for("nonexistent").is_none());
}

#[test]
fn attribution_report_dominant_cause_is_co2() {
    let mut g = simple_chain();
    g.forward_pass().unwrap();
    let report = CausalAttributor::attribute(&g, "temp").unwrap();
    let dominant = report.dominant_cause();
    assert!(dominant.is_some());
    assert_eq!(dominant.unwrap(), "co2");
}

#[test]
fn attributor_on_missing_target_returns_empty_weights() {
    let mut g = simple_chain();
    g.forward_pass().unwrap();
    let report = CausalAttributor::attribute(&g, "nonexistent").unwrap();
    assert!(
        report.weights.is_empty(),
        "expected empty weights for unknown target, got {:?}",
        report.weights
    );
}

#[test]
fn hypothesis_generator_produces_hypotheses_for_unexplained_node() {
    let mut g = CausalGraph::new();
    g.add_node("co2", Some(400.0));
    g.add_node("solar", Some(200.0));
    g.add_node("temp", None);
    g.add_edge("co2", "temp", Some(0.01)).unwrap();

    let mut observed = HashMap::new();
    observed.insert("temp".to_string(), 10.0);

    let generation = HypothesisGenerator::new(0.1, 5);
    let hypotheses = generation.generate(&g, &observed).unwrap();
    assert!(
        !hypotheses.is_empty(),
        "expected at least one hypothesis: solar→temp should be a candidate"
    );
}

#[test]
fn hypothesis_generator_respects_max_hypotheses_cap() {
    let mut g = CausalGraph::new();
    g.add_node("a", Some(1.0));
    g.add_node("b", Some(2.0));
    g.add_node("c", Some(3.0));
    g.add_node("target", None);
    g.add_edge("a", "target", Some(0.1)).unwrap();

    let mut observed = HashMap::new();
    observed.insert("target".to_string(), 100.0);

    let max = 2;
    let generation = HypothesisGenerator::new(0.01, max);
    let hypotheses = generation.generate(&g, &observed).unwrap();
    assert!(
        hypotheses.len() <= max,
        "hypotheses ({}) exceeded max ({max})",
        hypotheses.len()
    );
}

#[test]
fn hypothesis_generator_sorted_by_explanatory_power_descending() {
    let mut g = CausalGraph::new();
    g.add_node("strong", Some(5.0));
    g.add_node("weak", Some(0.5));
    g.add_node("target", None);
    g.add_edge("strong", "target", Some(1.0)).unwrap();

    let mut observed = HashMap::new();
    observed.insert("target".to_string(), 20.0);

    let generation = HypothesisGenerator::new(0.01, 10);
    let hypotheses = generation.generate(&g, &observed).unwrap();

    for i in 1..hypotheses.len() {
        assert!(
            hypotheses[i - 1].explanatory_power >= hypotheses[i].explanatory_power,
            "hypotheses not sorted: [{}]={} < [{}]={}",
            i - 1,
            hypotheses[i - 1].explanatory_power,
            i,
            hypotheses[i].explanatory_power
        );
    }
}

#[test]
fn hypothesis_proposed_edge_has_non_empty_names() {
    let g = simple_chain();
    let mut observed = HashMap::new();
    observed.insert("temp".to_string(), 100.0);

    let generation = HypothesisGenerator::new(0.01, 5);
    let hypotheses = generation.generate(&g, &observed).unwrap();
    for h in &hypotheses {
        assert!(!h.proposed_edge.from.is_empty());
        assert!(!h.proposed_edge.to.is_empty());
    }
}

#[test]
fn hypothesis_does_not_propose_existing_edges() {
    let g = simple_chain();
    let mut observed = HashMap::new();
    observed.insert("temp".to_string(), 100.0);

    let generation = HypothesisGenerator::new(0.01, 10);
    let hypotheses = generation.generate(&g, &observed).unwrap();

    for h in &hypotheses {
        let already_parent = g
            .parents(&h.proposed_edge.to)
            .contains(&h.proposed_edge.from);
        assert!(
            !already_parent,
            "hypothesis proposes existing edge {} → {}",
            h.proposed_edge.from, h.proposed_edge.to
        );
    }
}
