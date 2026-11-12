// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.

//! # Physics Simulation Command Handlers
//!
//! Handlers for physics simulation operations.

use crate::{
    cli::display,
    error::Result as LmmResult,
    physics::{HarmonicOscillator, LorenzSystem, Pendulum, SIRModel},
    simulation::Simulator,
    traits::Simulatable,
};
use tracing::info;

pub fn handle_physics(model: String, steps: usize, step_size: f64, verbose: bool) -> LmmResult<()> {
    display::banner("Physics · Dynamical Systems", "⚛️");
    let sim = Simulator { step_size };
    match model.as_str() {
        "lorenz" => {
            let sys = LorenzSystem::canonical()?;
            let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
            info!("✅ Lorenz: {} steps", traj.len() - 1);
            let final_data = &traj.last().unwrap().data;
            if verbose {
                display::divider("Final xyz");
                info!("  {:?}", final_data);
            } else {
                println!("{:?}", final_data);
            }
        }
        "pendulum" => {
            let sys = Pendulum::new(9.81, 1.0, 0.3, 0.0)?;
            let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
            info!("✅ Pendulum: {} steps", traj.len() - 1);
            let final_data = &traj.last().unwrap().data;
            if verbose {
                display::divider("Final [θ, ω]");
                info!("  {:?}", final_data);
            } else {
                println!("{:?}", final_data);
            }
        }
        "sir" => {
            let sys = SIRModel::new(0.3, 0.1, 990.0, 10.0, 0.0)?;
            let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
            info!("✅ SIR: {} steps", traj.len() - 1);
            let final_data = &traj.last().unwrap().data;
            if verbose {
                display::divider("Final [S, I, R]");
                info!("  {:?}", final_data);
            } else {
                println!("{:?}", final_data);
            }
        }
        _ => {
            let sys = HarmonicOscillator::new(1.0, 1.0, 0.0)?;
            let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
            info!("✅ Harmonic: {} steps", traj.len() - 1);
            let s = traj.last().unwrap();
            let energy = 0.5 * s.data[1] * s.data[1] + 0.5 * s.data[0] * s.data[0];
            if verbose {
                display::divider("Final [x, v]");
                info!("  {:?}", s.data);
                display::divider("Final Energy");
                info!("  {:.6}", energy);
            } else {
                println!("{:?}\n{:.6}", s.data, energy);
            }
        }
    }
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.
