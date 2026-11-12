// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.

//! # Causal Graph Command Handlers
//!
//! Handlers for causal graph operations.

use crate::{causal::CausalGraph, cli::display, error::Result as LmmResult, traits::Causal};
use tracing::info;

pub fn handle_causal(intervene_node: String, intervene_value: f64, verbose: bool) -> LmmResult<()> {
    display::banner("Causal · Intervention Engine", "🔀");
    let mut graph = CausalGraph::new();
    graph.add_node("x", Some(0.0));
    graph.add_node("y", None);
    graph.add_node("z", None);
    graph.add_edge("x", "y", Some(2.0))?;
    graph.add_edge("y", "z", Some(1.0))?;
    graph
        .nodes
        .iter_mut()
        .find(|n| n.name == "x")
        .unwrap()
        .value = Some(3.0);
    graph.forward_pass()?;
    display::divider("Before Intervention");
    info!(
        "  x={:?}  y={:?}  z={:?}",
        graph.get_value("x"),
        graph.get_value("y"),
        graph.get_value("z")
    );
    graph.intervene(&intervene_node, intervene_value)?;
    graph.forward_pass()?;
    if verbose {
        display::divider(&format!("After do({}={})", intervene_node, intervene_value));
        info!(
            "  x={:?}  y={:?}  z={:?}",
            graph.get_value("x"),
            graph.get_value("y"),
            graph.get_value("z")
        );
    } else {
        println!(
            "x={:?} y={:?} z={:?}",
            graph.get_value("x"),
            graph.get_value("y"),
            graph.get_value("z")
        );
    }
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.
