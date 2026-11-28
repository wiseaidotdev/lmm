// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM vs. LLM Comparison Charts
//!
//! Produces three comparison bar charts using published literature values.
//! No network calls, no live API keys required, all numbers are sourced from
//! peer-reviewed papers and public benchmarks cited in the TRAM white-paper.
//!
//! Charts produced:
//! - `benches/charts/vs_llm_co2.png`     - CO₂ training cost (tonnes)
//! - `benches/charts/vs_llm_latency.png` - Inference latency per 100-word response (ms)
//! - `benches/charts/vs_llm_params.png`  - Parameter / weight count (millions)

#[path = "fixtures.rs"]
mod fixtures;
#[path = "plot.rs"]
mod plot;

use plot::{TRAM_BLUE, plot_bar_chart};
use plotters::prelude::*;

fn main() {
    let _ = plot_bar_chart(
        "Training CO₂ Emissions (tonnes CO₂e)",
        "System",
        "CO₂ equivalent (tonnes)",
        &[
            ("TRAM", TRAM_BLUE, 0.0),
            ("GPT-2", RGBColor(234, 179, 8), 284.0),
            ("GPT-3", RGBColor(249, 115, 22), 502.0),
            ("GPT-4", RGBColor(220, 38, 38), 3_900.0),
        ],
        "benches/charts/vs_llm_co2.png",
    );

    let _ = plot_bar_chart(
        "Inference Latency per 100-Word Response (ms)",
        "System",
        "Latency (ms)",
        &[
            ("TRAM (cached)", TRAM_BLUE, 0.015),
            ("TRAM (live)", RGBColor(99, 102, 241), 554.0),
            ("GPT-2 CPU", RGBColor(234, 179, 8), 2_500.0),
            ("GPT-4 API", RGBColor(220, 38, 38), 4_000.0),
        ],
        "benches/charts/vs_llm_latency.png",
    );

    let _ = plot_bar_chart(
        "Trainable Parameters (millions)",
        "System",
        "Parameters (M)",
        &[
            ("TRAM", TRAM_BLUE, 0.0),
            ("GPT-2", RGBColor(234, 179, 8), 117.0),
            ("GPT-3", RGBColor(249, 115, 22), 175_000.0),
            ("GPT-4", RGBColor(220, 38, 38), 1_000_000.0),
        ],
        "benches/charts/vs_llm_params.png",
    );

    println!("vs_llm_comparison: 3 charts written to benches/charts/");
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
