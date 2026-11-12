// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Discover Command Handler

use crate::{cli::display, discovery::SymbolicRegression, error::Result as LmmResult};

pub fn handle_discover(iterations: usize, verbose: bool) -> LmmResult<()> {
    display::banner("Discover · Symbolic Regression", "🔭");
    let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
    let inputs: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
    let sr = SymbolicRegression::new(3, iterations).with_variables(vec!["x".into()]);
    let expr = sr.fit(&inputs, &ys)?;
    if verbose {
        display::divider("Discovered Equation");
        tracing::info!("  f(x) = {}", expr);
    } else {
        println!("{}", expr);
    }
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
