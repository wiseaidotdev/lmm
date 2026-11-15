// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Simulation Command Handlers

#[cfg(feature = "cli")]
use {
    crate::{
        cli::{
            display::{banner, divider},
            handlers::{
                causal::handle_causal, consciousness::handle_consciousness,
                discover::handle_discover, simulate::handle_simulate,
            },
        },
        field::Field,
        physics::{HarmonicOscillator, LorenzSystem, Pendulum, SIRModel},
        simulation::Simulator,
        tensor::Tensor,
        traits::Simulatable,
    },
    tracing::info,
};

/// Dispatches the `physics` subcommand to the appropriate dynamical-system simulator.
///
/// # Arguments
///
/// * `model`     - Model name: `"lorenz"`, `"pendulum"`, `"sir"`, or `"harmonic"` (default).
/// * `steps`     - Number of simulation steps.
/// * `step_size` - Integration step size.
/// * `verbose`   - When `true`, displays final state via `info!`; otherwise prints to stdout.
#[cfg(feature = "cli")]
pub fn handle_physics(
    model: String,
    steps: usize,
    step_size: f64,
    verbose: bool,
) -> anyhow::Result<()> {
    banner("Physics · Dynamical Systems", "⚛️");
    let sim = Simulator { step_size };
    match model.as_str() {
        "lorenz" => {
            let sys = LorenzSystem::canonical()?;
            let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
            info!("✅ Lorenz: {} steps", traj.len() - 1);
            let final_data = &traj.last().unwrap().data;
            if verbose {
                divider("Final xyz");
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
                divider("Final [θ, ω]");
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
                divider("Final [S, I, R]");
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
                divider("Final [x, v]");
                info!("  {:?}", s.data);
                divider("Final Energy");
                info!("  {:.6}", energy);
            } else {
                println!("{:?}\n{:.6}", s.data, energy);
            }
        }
    }
    Ok(())
}

/// Dispatches the `field` subcommand to the appropriate differential-operator computation.
///
/// # Arguments
///
/// * `size`      - Number of field elements (1-D).
/// * `operation` - Operation name: `"laplacian"` or `"gradient"` (default).
/// * `verbose`   - When `true`, displays result via `info!`; otherwise prints to stdout.
#[cfg(feature = "cli")]
pub fn handle_field(size: usize, operation: String, verbose: bool) -> anyhow::Result<()> {
    banner("Field · Differential Operators", "🌐");
    let data: Vec<f64> = (0..size).map(|i| (i as f64).powi(2)).collect();
    let tensor = Tensor::new(vec![size], data)?;
    let field = Field::new(vec![size], tensor)?;
    match operation.as_str() {
        "laplacian" => {
            let lap = field.compute_laplacian()?;
            if verbose {
                divider("∇² f(x)");
                info!("  {:?}", lap.values.data);
            } else {
                println!("{:?}", lap.values.data);
            }
        }
        _ => {
            let grad = field.compute_gradient()?;
            if verbose {
                divider("∇ f(x)");
                info!("  {:?}", grad.values.data);
            } else {
                println!("{:?}", grad.values.data);
            }
        }
    }
    Ok(())
}

/// Delegates the `simulate` subcommand to its CLI handler.
#[cfg(feature = "cli")]
pub fn dispatch_simulate(step: f64, steps: usize, verbose: bool) -> anyhow::Result<()> {
    Ok(handle_simulate(step, steps, verbose)?)
}

/// Delegates the `discover` subcommand to its CLI handler.
#[cfg(feature = "cli")]
pub fn dispatch_discover(iterations: usize, verbose: bool) -> anyhow::Result<()> {
    Ok(handle_discover(iterations, verbose)?)
}

/// Delegates the `consciousness` subcommand to its CLI handler.
#[cfg(feature = "cli")]
pub fn dispatch_consciousness(lookahead: usize, verbose: bool) -> anyhow::Result<()> {
    Ok(handle_consciousness(lookahead, verbose)?)
}

/// Delegates the `causal` subcommand to its CLI handler.
#[cfg(feature = "cli")]
pub fn dispatch_causal(
    intervene_node: String,
    intervene_value: f64,
    verbose: bool,
) -> anyhow::Result<()> {
    Ok(handle_causal(intervene_node, intervene_value, verbose)?)
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
