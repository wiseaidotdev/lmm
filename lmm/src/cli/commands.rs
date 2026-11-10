// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # CLI Command Definitions
//!
//! Defines the [`Cli`] top-level parser and the [`Commands`] enum covering every
//! subcommand exposed by the `lmm` binary. Each subcommand variant carries its
//! own typed argument set that is forwarded to the appropriate engine function
//! in [`crate::app`].
//!
//! The CLI is built with [`clap`] using the derive macro approach.
//!
//! ## Default mode
//!
//! When the user passes `-p 'question'` **without** a subcommand, `lmm`
//! automatically routes to `tram`, the Text Reasoning and Applied Mathematics
//! engine.

use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};

fn styles() -> Styles {
    clap::builder::styling::Styles::styled()
        .header(AnsiColor::Red.on_default() | Effects::BOLD)
        .usage(AnsiColor::Red.on_default() | Effects::BOLD)
        .literal(AnsiColor::Blue.on_default() | Effects::BOLD)
        .error(AnsiColor::Red.on_default() | Effects::BOLD)
        .placeholder(AnsiColor::Green.on_default())
}

/// Top-level argument parser for the `lmm` binary.
///
/// `command` is optional: when absent and `prompt` is set, `run_cli_entry`
/// defaults to the [`Commands::Tram`] subcommand.
#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    name = "lmm",
    propagate_version = true,
    styles = styles(),
    help_template = r#"{before-help}{name} {version}
{about}
{usage-heading} {usage}
{all-args}{after-help}

AUTHORS:
    {author}
"#,
    about = r#"
  ▄▄▄      ▄▄▄     ▄▄▄   ▄▄▄     ▄▄▄  
 ▀██▀       ███▄ ▄███     ███▄ ▄███   
  ██        ██ ▀█▀ ██     ██ ▀█▀ ██   
  ██        ██     ██     ██     ██   
  ██        ██     ██     ██     ██   
 ████████ ▀██▀     ▀██▄ ▀██▀     ▀██▄

Large Mathematical Model · Equation-Based Intelligence

Pass -p 'your question' without a subcommand to invoke TRAM mode directly.
"#
)]
pub struct Cli {
    #[arg(
        short,
        long,
        global = true,
        help = "Show detailed output with formatting"
    )]
    pub verbose: bool,

    /// Short-circuit prompt: invoke TRAM mode without spelling out the subcommand.
    ///
    /// `lmm -p 'what is entropy'` is equivalent to `lmm tram -p 'what is entropy'`.
    #[arg(
        short = 'p',
        long,
        global = false,
        help = "Prompt for TRAM mode (default subcommand when no subcommand is given)"
    )]
    pub prompt: Option<String>,

    /// Optional path to a local knowledge-base file or directory to ingest.
    ///
    /// Applied when using the top-level `-p` shortcut (default TRAM mode).
    #[arg(
        short = 'k',
        long,
        alias = "kb",
        global = false,
        help = "Optional path to a local knowledge-base file or directory for TRAM mode"
    )]
    pub knowledge_base: Option<String>,

    /// Per-word stochastic synonym-substitution probability [0.0, 1.0].
    ///
    /// Applied when using the top-level `-p` shortcut (default TRAM mode).
    /// Lower values produce less variation; `0.0` disables substitution entirely.
    #[arg(
        long,
        global = false,
        help = "Stochastic synonym-substitution probability [0.0, 1.0] for TRAM mode (default: 0.05 for KB, 0.12 for symbolic)"
    )]
    pub probability: Option<f64>,

    /// ThinkLoop reasoning iteration budget.
    ///
    /// Applied when using the top-level `-p` shortcut (default TRAM mode).
    #[arg(
        short = 'i',
        long,
        global = false,
        default_value_t = 10,
        help = "ThinkLoop reasoning iteration budget for TRAM mode (default: 10)"
    )]
    pub iterations: usize,

    /// GP expression-tree depth limit for TRAM mode.
    ///
    /// Applied when using the top-level `-p` shortcut (default TRAM mode).
    #[arg(
        short = 'd',
        long,
        global = false,
        default_value_t = 3,
        help = "GP expression-tree depth limit for TRAM mode (default: 3)"
    )]
    pub depth: usize,

    /// Expert inference backward-chaining depth limit.
    ///
    /// Controls how many rule-application steps the `InferenceEngine` explores
    /// before giving up. Higher values find deeper conclusions but are slower.
    /// Applied when using the top-level `-p` shortcut (default TRAM mode).
    #[arg(
        long,
        alias = "inf-depth",
        global = false,
        default_value_t = 3,
        help = "Expert inference backward-chaining max depth (default: 3)"
    )]
    pub inference_depth: usize,

    /// Expert inference certainty-factor threshold [0.0, 1.0].
    ///
    /// A derived conclusion is accepted only when its combined Prospector
    /// certainty factor meets or exceeds this value. Lower values make the
    /// engine more permissive; higher values require stronger evidence.
    /// Applied when using the top-level `-p` shortcut (default TRAM mode).
    #[arg(
        long,
        alias = "cf",
        global = false,
        default_value_t = 0.20,
        help = "Expert inference certainty-factor threshold [0.0, 1.0] (default: 0.20)"
    )]
    pub cf_threshold: f64,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// All subcommands available in the `lmm` binary.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Text Reasoning and Applied Mathematics, the default reasoning engine.
    ///
    /// TRAM ingests an optional local knowledge base, runs a closed-loop
    /// `ThinkLoop` reasoning cycle via `lmm-agent`, generates a symbolic
    /// continuation with `TextPredictor`, and applies stochastic synonym
    /// substitution to guarantee a unique response on every invocation.
    /// All computation runs on a single CPU core, no GPU, no neural weights.
    ///
    /// When the `net` feature is enabled and no knowledge base is provided,
    /// TRAM falls back to a DuckDuckGo search to enrich the seed before
    /// generation.
    #[command(about = "Reason from first principles: KB → web fallback → symbolic generation")]
    Tram {
        /// Natural-language prompt or question.
        #[arg(short, long)]
        prompt: String,
        /// Optional path to a local knowledge-base file or directory to ingest.
        #[arg(short, long, alias = "kb")]
        knowledge_base: Option<String>,
        /// Per-word stochastic synonym-substitution probability [0.0, 1.0].
        #[arg(long)]
        probability: Option<f64>,
        /// ThinkLoop reasoning iteration budget.
        #[arg(short, long, default_value_t = 10)]
        iterations: usize,
        /// GP expression-tree depth limit.
        #[arg(short, long, default_value_t = 3)]
        depth: usize,
        /// Expert inference backward-chaining max depth.
        #[arg(long, alias = "inf-depth", default_value_t = 3)]
        inference_depth: usize,
        /// Expert inference certainty-factor threshold [0.0, 1.0].
        #[arg(long, alias = "cf", default_value_t = 0.30)]
        cf_threshold: f64,
    },
    #[command(about = "Simulate continuous logical pathways")]
    Simulate {
        #[arg(short, long, default_value_t = 0.01)]
        step: f64,
        #[arg(short = 't', long, default_value_t = 100)]
        steps: usize,
    },
    #[command(about = "Discover governing equations from data")]
    Discover {
        #[arg(short, long, default_value = "synthetic")]
        data_path: String,
        #[arg(short = 'i', long, default_value_t = 100)]
        iterations: usize,
    },
    #[command(about = "Evaluate conscious state coherence")]
    Consciousness {
        #[arg(short, long, default_value_t = 3)]
        lookahead: usize,
    },
    #[command(about = "Run harmonic and chaotic physical models")]
    Physics {
        #[arg(short, long, default_value = "harmonic")]
        model: String,
        #[arg(short, long, default_value_t = 200)]
        steps: usize,
        #[arg(short = 'z', long, default_value_t = 0.01)]
        step_size: f64,
    },
    #[command(about = "Perform causal interventions and counterfactuals")]
    Causal {
        #[arg(short = 'n', long, default_value = "x")]
        intervene_node: String,
        #[arg(short = 'v', long, default_value_t = 1.0)]
        intervene_value: f64,
    },
    #[command(about = "Compute tensor field gradients and divergences")]
    Field {
        #[arg(short, long, default_value_t = 10)]
        size: usize,
        #[arg(short = 'o', long, default_value = "gradient")]
        operation: String,
    },
    #[command(about = "Encode continuous truth into discrete text")]
    Encode {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(short, long, default_value = "Hello, LMM!")]
        text: String,
        #[arg(long, default_value_t = 80)]
        iterations: usize,
        #[arg(long, default_value_t = 4)]
        depth: usize,
    },
    #[command(about = "Decode text back into dynamic equations")]
    Decode {
        #[arg(short, long)]
        equation: String,
        #[arg(short, long)]
        length: usize,
        #[arg(short, long, default_value = "", allow_hyphen_values = true)]
        residuals: String,
    },
    #[command(about = "Predict next sequence based on pattern logic")]
    Predict {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(short, long, default_value = "The Pharaohs encoded reality in")]
        text: String,
        #[arg(short = 'w', long, default_value_t = 32)]
        window: usize,
        #[arg(short = 'p', long, default_value_t = 16)]
        predict_length: usize,
        #[arg(long, default_value_t = 80)]
        iterations: usize,
        #[arg(long, default_value_t = 4)]
        depth: usize,
        #[arg(long)]
        dictionary: Option<String>,
        #[arg(long, default_value_t = false)]
        stochastic: bool,
        #[arg(long, default_value_t = 0.5)]
        probability: f64,
    },
    #[command(about = "Extract key meaning via GP scoring")]
    Summarize {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(
            short,
            long,
            default_value = "Equations are the universe. Mathematical models compress reality into compact symbolic forms. Simulation is more powerful than description. The world is not made of words but of structure and force."
        )]
        text: String,
        #[arg(short = 'n', long, default_value_t = 2)]
        sentences: usize,
        #[arg(long, default_value_t = 40)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value_t = false)]
        stochastic: bool,
        #[arg(long, default_value_t = 0.5)]
        probability: f64,
    },
    #[command(about = "Generate a single structural sentence")]
    Sentence {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(short, long, default_value = "Mathematical equations")]
        text: String,
        #[arg(long, default_value_t = 60)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value_t = false)]
        stochastic: bool,
        #[arg(long, default_value_t = 0.5)]
        probability: f64,
    },
    #[command(about = "Generate a cohesive paragraph from a seed")]
    Paragraph {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(
            short,
            long,
            default_value = "Equations encode the structure of reality"
        )]
        text: String,
        #[arg(short = 'n', long, default_value_t = 3)]
        sentences: usize,
        #[arg(long, default_value_t = 60)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value_t = false)]
        stochastic: bool,
        #[arg(long, default_value_t = 0.5)]
        probability: f64,
    },
    #[command(about = "Structure a full essay with intro and conclusion")]
    Essay {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(
            short,
            long,
            default_value = "Mathematical models and the structure of reality"
        )]
        text: String,
        #[arg(short = 'n', long, default_value_t = 2)]
        paragraphs: usize,
        #[arg(short = 's', long, default_value_t = 3)]
        sentences: usize,
        #[arg(long, default_value_t = 60)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value_t = false)]
        stochastic: bool,
        #[arg(long, default_value_t = 0.5)]
        probability: f64,
    },
    #[cfg(feature = "net")]
    #[command(about = "Ask a question and get an equation-scored answer from the web")]
    Ask {
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
        #[arg(short = 'n', long, default_value_t = 3)]
        sentences: usize,
        #[arg(long, default_value = "wt-wt")]
        region: String,
        #[arg(long, default_value_t = 40)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value_t = false)]
        stochastic: bool,
        #[arg(long, default_value_t = 0.5)]
        probability: f64,
    },
    #[command(about = "Generate an image from text via Spectral Field Synthesis")]
    Imagen {
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value_t = 512)]
        width: u32,
        #[arg(long, default_value_t = 512)]
        height: u32,
        #[arg(short = 'c', long, default_value_t = 8)]
        components: usize,
        #[arg(short = 's', long, default_value = "plasma")]
        style: String,
        #[arg(long, default_value = "auto")]
        palette: String,
        #[arg(short = 'o', long, default_value = "output.ppm")]
        output: String,
    },
}
