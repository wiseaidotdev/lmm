// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.

//! # Consciousness Simulation Command Handlers
//!
//! Handlers for consciousness simulation operations.

use crate::{
    cli::display, consciousness::Consciousness, error::Result as LmmResult, tensor::Tensor,
};
use tracing::info;

pub fn handle_consciousness(lookahead: usize, verbose: bool) -> LmmResult<()> {
    display::banner("Consciousness · Perception Loop", "🧠");
    let mut consc = Consciousness::new(Tensor::zeros(vec![4]), lookahead, 0.01);
    let input = vec![128u8, 64, 32, 255];
    let state = consc.tick(&input)?;
    if verbose {
        display::divider("State");
        info!("  {:?}", state.data);
        display::divider("Mean Prediction Error");
        info!("  {:.6}", consc.mean_prediction_error());
    } else {
        println!("{:?}\n{:.6}", state.data, consc.mean_prediction_error());
    }
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.
