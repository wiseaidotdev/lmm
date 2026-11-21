// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Generation Command Handlers

#[cfg(all(feature = "cli", feature = "net"))]
use {
    crate::{
        app::helpers::maybe_enhance,
        cli::display::{banner, divider},
        net::{
            SearchAggregator, corpus_from_response, corpus_from_results, corpus_from_results_raw,
        },
        text::TextSummarizer,
    },
    tracing::{error, info},
};

#[cfg(all(feature = "cli", not(feature = "net")))]
use {
    crate::{
        cli::display::{banner, divider},
        imagen::{ImagenParams, StyleMode, render as imagen_render},
    },
    tracing::{error, info},
};

#[cfg(all(feature = "cli", feature = "net"))]
use crate::imagen::{ImagenParams, StyleMode, render as imagen_render};

/// Configuration for the `ask` command.
#[cfg(all(feature = "cli", feature = "net"))]
pub struct AskConfig {
    /// Natural-language question.
    pub prompt: String,
    /// Maximum number of search results to fetch.
    pub limit: usize,
    /// Number of sentences to extract in the summary.
    pub sentences: usize,
    /// DuckDuckGo region code (e.g. `"wt-wt"`).
    pub region: String,
    /// Summariser GP iterations.
    pub iterations: usize,
    /// Summariser recursion depth.
    pub depth: usize,
    /// Whether to apply synonym substitution.
    pub stochastic: bool,
    /// Substitution probability.
    pub probability: f64,
    /// Verbose output flag.
    pub verbose: bool,
}

/// Handles the `ask` subcommand, internet-aware knowledge synthesis via DuckDuckGo.
#[cfg(all(feature = "cli", feature = "net"))]
pub async fn handle_ask(cfg: AskConfig) -> anyhow::Result<()> {
    let AskConfig {
        prompt,
        limit,
        sentences,
        region,
        iterations,
        depth,
        stochastic,
        probability,
        verbose,
    } = cfg;
    banner("Ask · Internet-Aware Knowledge Synthesis", "🌐");
    info!("  ❓ Prompt : {:?}", prompt);
    let aggregator = SearchAggregator::new().with_region(&region);
    divider("DuckDuckGo Results");

    if let Err(e) = aggregator.search_and_display(&prompt, limit).await {
        error!("  ⚠️  Search display failed: {}", e);
    }

    let corpus = match aggregator.get_response(&prompt).await {
        Ok(resp) => corpus_from_response(&resp),
        Err(_) => String::new(),
    };

    let final_corpus = if corpus.trim().is_empty() {
        let lite_results = aggregator.fetch(&prompt, limit).await.unwrap_or_default();
        let quality = corpus_from_results(&lite_results);
        if quality.trim().is_empty() {
            corpus_from_results_raw(&lite_results)
        } else {
            quality
        }
    } else {
        corpus
    };

    info!("");
    divider("LMM Response");

    if final_corpus.trim().is_empty() {
        error!("  ❌ No extractable content from search results.");
    } else {
        let summarizer = TextSummarizer::new(sentences, iterations, depth);
        match summarizer.summarize_with_query(&final_corpus, &prompt) {
            Ok(summary) => {
                for sentence in &summary {
                    let enhanced = maybe_enhance(sentence, stochastic, probability);
                    if verbose {
                        info!("  {}", enhanced);
                    } else {
                        info!("{}", enhanced);
                    }
                }
            }
            Err(e) => error!("  ❌ Summarization failed: {}", e),
        }
    }
    Ok(())
}

/// Configuration for the `imagen` command.
#[cfg(feature = "cli")]
pub struct ImagenConfig {
    /// Textual description driving the spectral palette.
    pub prompt: String,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Number of Fourier components.
    pub components: usize,
    /// Style mode string (e.g. `"plasma"`, `"crystal"`).
    pub style: String,
    /// Named colour palette.
    pub palette: String,
    /// Output file path.
    pub output: String,
    /// Verbose output flag.
    pub verbose: bool,
}

/// Handles the `imagen` subcommand, spectral field image synthesis.
#[cfg(feature = "cli")]
pub fn handle_imagen(cfg: ImagenConfig) -> anyhow::Result<()> {
    let ImagenConfig {
        prompt,
        width,
        height,
        components,
        style,
        palette,
        output,
        verbose,
    } = cfg;
    banner("Imagen · Spectral Field Synthesis", "🎨");
    info!("  🖼  Prompt    : {:?}", prompt);
    info!("  📐 Dimensions : {}x{}", width, height);
    info!("  🎭 Style      : {}", style);
    info!("  🎨 Palette    : {}", palette);
    info!("  🌊 Components : {}", components);
    let parsed_style = style.parse::<StyleMode>().unwrap_or(StyleMode::Plasma);
    let params = ImagenParams {
        prompt,
        width,
        height,
        components,
        style: parsed_style,
        palette_name: palette,
        output,
    };
    info!("");
    divider("Rendering");
    match imagen_render(&params) {
        Ok(path) => {
            if verbose {
                info!("  ✅ Saved to: {}", path);
            } else {
                info!("{}", path);
            }
        }
        Err(e) => error!("  ❌ Render failed: {}", e),
    }
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
