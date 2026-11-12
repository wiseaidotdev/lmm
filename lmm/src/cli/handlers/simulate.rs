// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Simulate Command Handler

use crate::{
    cli::display, error::Result as LmmResult, physics::HarmonicOscillator, simulation::Simulator,
    traits::Simulatable,
};

pub fn handle_simulate(step: f64, steps: usize, verbose: bool) -> LmmResult<()> {
    display::banner("Simulate · Harmonic Oscillator", "🌊");
    let osc = HarmonicOscillator::new(1.0, 1.0, 0.0)?;
    let sim = Simulator { step_size: step };
    let trajectory = sim.simulate_trajectory(&osc, osc.state(), steps)?;
    tracing::info!(
        "✅ Simulation complete - {} steps, step_size={}",
        trajectory.len() - 1,
        step
    );
    let final_data = &trajectory.last().unwrap().data;
    if verbose {
        display::divider("Final State");
        tracing::info!("  {:?}", final_data);
    } else {
        println!("{:?}", final_data);
    }
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
