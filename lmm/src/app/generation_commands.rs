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
        net::{SearchAggregator, corpus_from_response, corpus_from_results, corpus_from_results_raw},
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

/// Handles the `ask` subcommand, internet-aware knowledge synthesis via DuckDuckGo.
///
/// # Arguments
///
/// * `prompt`     - Natural-language question.
/// * `limit`      - Maximum number of search results to fetch.
/// * `sentences`  - Number of sentences to extract in the summary.
/// * `region`     - DuckDuckGo region code (e.g. `"wt-wt"`).
/// * `iterations` - Summariser GP iterations.
/// * `depth`      - Summariser recursion depth.
/// * `stochastic` - Whether to apply synonym substitution.
/// * `probability`- Substitution probability.
/// * `verbose`    - Verbose output flag.
#[cfg(all(feature = "cli", feature = "net"))]
pub async fn handle_ask(
    prompt: String,
    limit: usize,
    sentences: usize,
    region: String,
    iterations: usize,
    depth: usize,
    stochastic: bool,
    probability: f64,
    verbose: bool,
) -> anyhow::Result<()> {
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
                        println!("{}", enhanced);
                    }
                }
            }
            Err(e) => error!("  ❌ Summarization failed: {}", e),
        }
    }
    Ok(())
}

/// Handles the `imagen` subcommand, spectral field image synthesis.
///
/// # Arguments
///
/// * `prompt`     - Textual description driving the spectral palette.
/// * `width`      - Output image width in pixels.
/// * `height`     - Output image height in pixels.
/// * `components` - Number of Fourier components.
/// * `style`      - Style mode string (e.g. `"plasma"`, `"crystal"`).
/// * `palette`    - Named colour palette.
/// * `output`     - Output file path.
/// * `verbose`    - Verbose output flag.
#[cfg(feature = "cli")]
pub fn handle_imagen(
    prompt: String,
    width: u32,
    height: u32,
    components: usize,
    style: String,
    palette: String,
    output: String,
    verbose: bool,
) -> anyhow::Result<()> {
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
                println!("{}", path);
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
