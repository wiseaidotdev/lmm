use lmm::causal::CausalGraph;
use lmm::traits::Causal;
use std::collections::HashMap;

fn build_chain() -> CausalGraph {
    let mut g = CausalGraph::new();
    g.add_node("x", None);
    g.add_node("y", None);
    g.add_node("z", None);
    g.add_edge("x", "y", None).unwrap();
    g.add_edge("y", "z", None).unwrap();
    g.nodes.iter_mut().find(|n| n.name == "x").unwrap().value = Some(3.0);
    g
}

#[test]
fn test_forward_pass() {
    let mut g = build_chain();
    g.forward_pass().unwrap();
    assert!((g.get_value("x").unwrap() - 3.0).abs() < 1e-10);
    assert!((g.get_value("y").unwrap() - 3.0).abs() < 1e-10);
    assert!((g.get_value("z").unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_intervention_freezes_node() {
    let mut g = build_chain();
    g.intervene("y", 10.0).unwrap();
    g.forward_pass().unwrap();
    assert!(
        (g.get_value("y").unwrap() - 10.0).abs() < 1e-10,
        "Intervened y should be 10"
    );
    assert!(
        (g.get_value("z").unwrap() - 10.0).abs() < 1e-10,
        "z = y (coeff=1)"
    );
}

#[test]
fn test_intervention_isolates_from_parent() {
    let mut g = build_chain();
    g.nodes.iter_mut().find(|n| n.name == "x").unwrap().value = Some(100.0);
    g.intervene("y", 5.0).unwrap();
    g.forward_pass().unwrap();
    assert!(
        (g.get_value("y").unwrap() - 5.0).abs() < 1e-10,
        "do(y=5) ignores x=100"
    );
}

#[test]
fn test_counterfactual() {
    let g = build_chain();
    let cf_z = g.counterfactual("x", 5.0, "z").unwrap();
    assert!((cf_z - 5.0).abs() < 1e-10, "chain with coeff=1 → z=x=5");
}

#[test]
fn test_parents_children() {
    let g = build_chain();
    assert_eq!(g.parents("y"), vec!["x".to_string()]);
    assert_eq!(g.children("y"), vec!["z".to_string()]);
    assert!(g.parents("x").is_empty());
    assert!(g.children("z").is_empty());
}

#[test]
fn test_markov_blanket() {
    let g = build_chain();
    let blanket = g.markov_blanket("y");
    assert!(
        blanket.contains("x"),
        "Parent x should be in Markov blanket of y"
    );
    assert!(
        blanket.contains("z"),
        "Child z should be in Markov blanket of y"
    );
}

#[test]
fn test_topological_order() {
    let g = build_chain();
    let order = g.topological_order().unwrap();
    let pos: HashMap<String, usize> = order
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    assert!(pos["x"] < pos["y"], "x before y");
    assert!(pos["y"] < pos["z"], "y before z");
}

#[test]
fn test_cycle_detection() {
    let mut g = CausalGraph::new();
    g.add_node("a", None);
    g.add_node("b", None);
    g.add_edge("a", "b", None).unwrap();
    g.add_edge("b", "a", None).unwrap();
    assert!(g.has_cycle());
}

#[test]
fn test_no_cycle_in_dag() {
    let g = build_chain();
    assert!(!g.has_cycle());
}

#[test]
fn test_invalid_edge_error() {
    let mut g = CausalGraph::new();
    g.add_node("a", None);
    assert!(g.add_edge("a", "nonexistent", None).is_err());
}

#[test]
fn test_intervene_unknown_node_error() {
    let mut g = CausalGraph::new();
    g.add_node("a", None);
    assert!(g.intervene("nonexistent", 1.0).is_err());
}
