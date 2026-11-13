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
        default_value_t = 5,
        help = "Expert inference backward-chaining max depth (default: 5)"
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
        default_value_t = 0.25,
        help = "Expert inference certainty-factor threshold [0.0, 1.0] (default: 0.25)"
    )]
    pub cf_threshold: f64,

    /// Enable the Adaptive Thought Engine (ATE) reasoning pre-pass.
    ///
    /// When set, ATE runs before `ThinkLoop`, classifies the prompt into one
    /// of seven reasoning modes, decomposes the goal into sub-goals, and
    /// feeds prioritised queries downstream. Off by default to keep latency
    /// low for simple look-ups; enable when deeper reasoning is desired.
    ///
    /// Applied when using the top-level `-p` shortcut (default TRAM mode).
    #[arg(
        short = 't',
        long = "think",
        global = false,
        default_value_t = false,
        help = "Enable the Adaptive Thought Engine reasoning pre-pass (default: off)"
    )]
    pub think: bool,

    /// Maximum number of internal thought steps the ATE will generate.
    ///
    /// Each step queries the oracle once; higher values produce richer
    /// reasoning chains but increase latency proportionally.
    /// Only used when `--think` is active.
    #[arg(
        long,
        global = false,
        default_value_t = 8,
        help = "Max ATE thought steps (default: 8, only used with --think)"
    )]
    pub think_steps: usize,

    /// DS-belief midpoint threshold for ATE early-stop convergence [0.0, 1.0].
    ///
    /// When the running Dempster-Shafer combined belief exceeds this value,
    /// the thought chain stops early. Lower values allow more exploration;
    /// values near 1.0 force the full `--think-steps` to run.
    /// Only used when `--think` is active.
    #[arg(
        long,
        global = false,
        default_value_t = 0.40,
        help = "ATE belief threshold for early-stop [0.0, 1.0] (default: 0.40, only with --think)"
    )]
    pub think_threshold: f64,

    /// Consecutive non-improving steps before the ATE critic flags stagnation.
    ///
    /// When belief has not improved by the minimum epsilon for this many
    /// consecutive steps, a corrective thought step is injected. Must be ≥ 1.
    /// Only used when `--think` is active.
    #[arg(
        long,
        global = false,
        default_value_t = 3,
        help = "ATE critic stagnation patience in steps (default: 3, only with --think)"
    )]
    pub think_patience: usize,

    /// Maximum number of intent-specific sub-tasks the pipeline will fire.
    ///
    /// Each sub-task issues an additional web query aimed at the specific
    /// intent of the prompt (advantages, comparison, code example, etc.).
    /// Higher values enrich the response at the cost of extra latency.
    #[arg(
        long,
        global = false,
        default_value_t = 4,
        help = "Max intent-specific sub-tasks for web enrichment (default: 4)"
    )]
    pub max_tasks: usize,

    /// Bypass response framing and emit raw pipeline output.
    ///
    /// By default the final answer is prefixed with an intent-specific opening
    /// phrase (e.g. "The advantages of Rust are...").  Pass `--no-frame` to
    /// suppress that prefix and receive the raw shuffled prose.
    #[arg(
        long,
        global = false,
        default_value_t = false,
        help = "Suppress intent-specific response framing (default: off)"
    )]
    pub no_frame: bool,

    /// Enable deep web scraping for topic-relevant content.
    ///
    /// When enabled, the system will iteratively scrape websites from DuckDuckGo
    /// `!ducky` results to find content similar to the topic being asked,
    /// and inject this new knowledge corpus into the pipeline. This is useful
    /// when the Instant Answer API returns generic or repetitive content.
    #[arg(
        long,
        global = false,
        default_value_t = false,
        help = "Enable deep web scraping for topic-relevant content (default: off)"
    )]
    pub deep: bool,

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
        #[arg(long, alias = "inf-depth", default_value_t = 5)]
        inference_depth: usize,
        /// Expert inference certainty-factor threshold [0.0, 1.0].
        #[arg(long, alias = "cf", default_value_t = 0.25)]
        cf_threshold: f64,
        /// Enable the Adaptive Thought Engine reasoning pre-pass.
        ///
        /// Off by default; pass `--think` or `-t` to activate deeper reasoning.
        #[arg(short = 't', long = "think", default_value_t = false)]
        think: bool,
        /// Maximum number of internal thought steps the ATE will generate.
        ///
        /// Only effective when `--think` is active.
        #[arg(long, default_value_t = 8)]
        think_steps: usize,
        /// DS-belief midpoint threshold for ATE early-stop convergence [0.0, 1.0].
        ///
        /// Only effective when `--think` is active.
        #[arg(long, default_value_t = 0.40)]
        think_threshold: f64,
        /// Consecutive non-improving steps before the ATE critic flags stagnation.
        ///
        /// Only effective when `--think` is active.
        #[arg(long, default_value_t = 3)]
        think_patience: usize,
        /// Maximum number of intent-specific sub-tasks for web enrichment.
        #[arg(long, default_value_t = 4)]
        max_tasks: usize,
        /// Suppress intent-specific response framing and emit raw pipeline output.
        #[arg(long, default_value_t = false)]
        no_frame: bool,
        /// Enable deep web scraping for topic-relevant content.
        #[arg(long, default_value_t = false)]
        deep: bool,
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
