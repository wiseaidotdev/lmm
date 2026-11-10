// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/wiseaidotdev/lmm/main/assets/favicon.png",
    html_logo_url = "https://raw.githubusercontent.com/wiseaidotdev/lmm/main/assets/logo.png"
)]
#![doc = include_str!("../README.md")]
#![doc = include_str!("../AGENT.md")]
#![doc = include_str!("../DERIVE.md")]
#![doc = include_str!("../RUST.md")]
#![doc = include_str!("../WASM.md")]

pub mod causal;
#[cfg(feature = "cli")]
pub mod cli;
pub mod cognition;
pub mod compression;
pub mod consciousness;
pub mod discovery;
pub mod encode;
pub mod equation;
pub mod error;
pub mod expert;
pub mod field;
pub mod generation;
pub mod imagen;
pub mod lexicon;
pub mod models;
#[cfg(any(feature = "net", feature = "wasm-net"))]
pub mod net;
pub mod operator;
pub mod perception;
pub mod physics;
pub mod predict;
pub mod prelude;
pub mod reasoner;
pub mod simulation;
pub mod stochastic;
pub mod symbolic;
pub mod tensor;
pub mod text;
pub mod traits;
pub mod uncertainty;
pub mod world;

pub mod app;

#[cfg(all(feature = "python", not(feature = "rust-binary")))]
pub mod python;

#[cfg(all(feature = "python", not(feature = "rust-binary")))]
use {
    app::run_cli_entry, pyo3::prelude::*, python::register_python_module, std::io::Error as IoError,
};

#[cfg(all(feature = "python", not(feature = "rust-binary")))]
#[pyfunction]
fn run_cli(args: Vec<String>) -> PyResult<()> {
    tokio::runtime::Runtime::new()
        .map_err(|e: IoError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .block_on(run_cli_entry(args))
        .map_err(|e: anyhow::Error| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(all(feature = "python", not(feature = "rust-binary")))]
#[pymodule]
fn _lmm(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    register_python_module(py, m)?;
    Ok(())
}

#[cfg(all(feature = "node", not(feature = "rust-binary")))]
pub mod node;

#[cfg(all(feature = "node", not(feature = "rust-binary")))]
use {app::run_cli_entry, napi_derive::napi, std::process::exit as process_exit};

#[cfg(all(feature = "node", not(feature = "rust-binary")))]
#[napi]
pub fn run_cli(args: Vec<String>) {
    tokio::runtime::Runtime::new()
        .expect("tokio runtime")
        .block_on(async move {
            if let Err(e) = run_cli_entry(args).await {
                eprintln!("Error: {}", e);
                process_exit(1);
            }
        });
}
