// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Text-Generation Command Handlers

#[cfg(feature = "cli")]
use {
    crate::{
        app::helpers::{load_source, maybe_enhance},
        cli::display::{banner, divider},
        constants::app::{
            ESSAY_LABEL_BODY_PREFIX, ESSAY_LABEL_CONCLUSION, ESSAY_LABEL_INTRODUCTION,
        },
        lexicon::Lexicon,
        predict::TextPredictor,
        text::{EssayGenerator, ParagraphGenerator, SentenceGenerator, TextSummarizer},
    },
    std::path::Path,
    tracing::{error, info},
};

/// Configuration for the `predict` command.
#[cfg(feature = "cli")]
pub struct PredictConfig {
    /// File path to load as seed corpus.
    pub input: String,
    /// Fallback inline seed text.
    pub text: String,
    /// N-gram window width.
    pub window: usize,
    /// Number of words to predict.
    pub predict_length: usize,
    /// GP iterations.
    pub iterations: usize,
    /// Recursion depth.
    pub depth: usize,
    /// Optional path to a word lexicon.
    pub dictionary: Option<String>,
    /// Whether to apply synonym substitution.
    pub stochastic: bool,
    /// Substitution probability (used when `stochastic` is `true`).
    pub probability: f64,
    /// Verbose output flag.
    pub verbose: bool,
}

/// Handles the `predict` subcommand, symbolic text continuation.
#[cfg(feature = "cli")]
pub fn handle_predict(cfg: PredictConfig) -> anyhow::Result<()> {
    let PredictConfig {
        input,
        text,
        window,
        predict_length,
        iterations,
        depth,
        dictionary,
        stochastic,
        probability,
        verbose,
    } = cfg;
    banner("Predict · Symbolic Continuation", "🔮");
    let source = load_source(&input, text)?;
    let lexicon = match dictionary {
        Some(path) => Lexicon::load_from(Path::new(&path)).ok(),
        None => Lexicon::load_system().ok(),
    };
    let mut predictor = TextPredictor::new(window, iterations, depth);
    if let Some(lex) = lexicon {
        info!("  📚 Dictionary: {} words loaded", lex.word_count());
        predictor = predictor.with_lexicon(lex);
    }
    let result = predictor.predict_continuation(&source, predict_length)?;
    info!("  📝 Input     : {:?}", source);
    info!("  🪟 Window    : {} words", result.window_used);
    info!("  📈 Trajectory: {}", result.trajectory_equation);
    info!("  🎵 Rhythm    : {}", result.rhythm_equation);
    let cont = maybe_enhance(&result.continuation, stochastic, probability);
    if verbose {
        divider("Continuation");
        info!("  {}{}", source, cont);
    } else {
        info!("{}{}", source, cont);
    }
    Ok(())
}

/// Configuration for the `summarize` command.
#[cfg(feature = "cli")]
pub struct SummarizeConfig {
    /// File path to load as source corpus.
    pub input: String,
    /// Fallback inline corpus text.
    pub text: String,
    /// Number of sentences to extract.
    pub sentences: usize,
    /// GP iterations.
    pub iterations: usize,
    /// Recursion depth.
    pub depth: usize,
    /// Whether to apply synonym substitution.
    pub stochastic: bool,
    /// Substitution probability.
    pub probability: f64,
    /// Verbose output flag.
    pub verbose: bool,
}

/// Handles the `summarize` subcommand, extractive key-sentence summarisation.
#[cfg(feature = "cli")]
pub fn handle_summarize(cfg: SummarizeConfig) -> anyhow::Result<()> {
    let SummarizeConfig {
        input,
        text,
        sentences,
        iterations,
        depth,
        stochastic,
        probability,
        verbose,
    } = cfg;
    banner("Summarize · Key Sentence Extraction", "✂️");
    let source = load_source(&input, text)?;
    info!("  📝 Input : {} chars", source.len());
    info!(
        "  📊 Extracting {} key sentences via GP scoring...",
        sentences
    );
    let summarizer = TextSummarizer::new(sentences, iterations, depth);
    match summarizer.summarize(&source) {
        Ok(summary) => {
            if verbose {
                divider("Summary");
            }
            for (i, sentence) in summary.iter().enumerate() {
                let enhanced = maybe_enhance(sentence, stochastic, probability);
                if verbose {
                    info!("  {}. {}", i + 1, enhanced);
                } else {
                    info!("{}", enhanced);
                }
            }
        }
        Err(e) => {
            error!("  ❌ Summarization failed: {}", e);
        }
    }
    Ok(())
}

/// Handles the `sentence` subcommand, single symbolic sentence generation.
///
/// # Arguments
///
/// * `input`      - File path to load as seed.
/// * `text`       - Fallback inline seed text.
/// * `iterations` - GP iterations.
/// * `depth`      - Recursion depth.
/// * `stochastic` - Whether to apply synonym substitution.
/// * `probability`- Substitution probability.
/// * `verbose`    - Verbose output flag.
#[cfg(feature = "cli")]
pub fn handle_sentence(
    input: String,
    text: String,
    iterations: usize,
    depth: usize,
    stochastic: bool,
    probability: f64,
    verbose: bool,
) -> anyhow::Result<()> {
    banner("Sentence · Single Sentence Generation", "✍️");
    let source = load_source(&input, text)?;
    info!("  🌱 Seed : {:?}", source);
    let sentence_gen = SentenceGenerator::new(iterations, depth);
    match sentence_gen.generate(&source) {
        Ok(sentence) => {
            let enhanced = maybe_enhance(&sentence, stochastic, probability);
            if verbose {
                divider("Generated Sentence");
                info!("  {}", enhanced);
            } else {
                info!("{}", enhanced);
            }
        }
        Err(e) => {
            error!("  ❌ Generation failed: {}", e);
        }
    }
    Ok(())
}

/// Configuration for the `paragraph` command.
#[cfg(feature = "cli")]
pub struct ParagraphConfig {
    /// File path to load as seed.
    pub input: String,
    /// Fallback inline seed text.
    pub text: String,
    /// Number of sentences in the paragraph.
    pub sentences: usize,
    /// GP iterations.
    pub iterations: usize,
    /// Recursion depth.
    pub depth: usize,
    /// Whether to apply synonym substitution.
    pub stochastic: bool,
    /// Substitution probability.
    pub probability: f64,
    /// Verbose output flag.
    pub verbose: bool,
}

/// Handles the `paragraph` subcommand, multi-sentence symbolic paragraph generation.
#[cfg(feature = "cli")]
pub fn handle_paragraph(cfg: ParagraphConfig) -> anyhow::Result<()> {
    let ParagraphConfig {
        input,
        text,
        sentences,
        iterations,
        depth,
        stochastic,
        probability,
        verbose,
    } = cfg;
    banner("Paragraph · Generate a Paragraph", "📄");
    let source = load_source(&input, text)?;
    info!("  🌱 Seed      : {:?}", source);
    info!("  📊 Sentences : {}", sentences);
    let para_gen = ParagraphGenerator::new(sentences, iterations, depth);
    match para_gen.generate(&source) {
        Ok(paragraph) => {
            let enhanced = maybe_enhance(&paragraph, stochastic, probability);
            if verbose {
                divider("Generated Paragraph");
                info!("");
                info!("  {}", enhanced);
            } else {
                info!("{}", enhanced);
            }
        }
        Err(e) => {
            error!("  ❌ Generation failed: {}", e);
        }
    }
    Ok(())
}

/// Configuration for the `essay` command.
#[cfg(feature = "cli")]
pub struct EssayConfig {
    /// File path to load as topic seed.
    pub input: String,
    /// Fallback inline topic text.
    pub text: String,
    /// Number of paragraphs to generate.
    pub paragraphs: usize,
    /// Number of sentences per paragraph.
    pub sentences: usize,
    /// GP iterations.
    pub iterations: usize,
    /// Recursion depth.
    pub depth: usize,
    /// Whether to apply synonym substitution.
    pub stochastic: bool,
    /// Substitution probability (used when `stochastic` is `true`).
    pub probability: f64,
    /// Verbose output flag.
    pub verbose: bool,
}

/// Handles the `essay` subcommand, multi-paragraph symbolic essay generation.
#[cfg(feature = "cli")]
pub fn handle_essay(cfg: EssayConfig) -> anyhow::Result<()> {
    let EssayConfig {
        input,
        text,
        paragraphs,
        sentences,
        iterations,
        depth,
        stochastic,
        probability,
        verbose,
    } = cfg;
    banner("Essay · Generate a Full Essay", "📖");
    let source = load_source(&input, text)?;
    info!("  🌱 Topic      : {:?}", source);
    info!(
        "  📊 Paragraphs : {} ({} sentences each)",
        paragraphs, sentences
    );
    let essay_gen = EssayGenerator::new(paragraphs, sentences, iterations, depth);
    match essay_gen.generate(&source) {
        Ok(essay) => {
            if verbose {
                divider("Essay");
                info!("");
                info!("  ══════════════════════════════════════");
                info!("  📖  {}", essay.title);
                info!("  ══════════════════════════════════════");
            } else {
                info!("{}\n", essay.title);
            }
            let count = essay.paragraphs.len();
            for (i, para) in essay.paragraphs.iter().enumerate() {
                let enhanced = maybe_enhance(para, stochastic, probability);
                if verbose {
                    let label = if i == 0 {
                        ESSAY_LABEL_INTRODUCTION.to_string()
                    } else if i == count - 1 {
                        ESSAY_LABEL_CONCLUSION.to_string()
                    } else {
                        format!("{}{}", ESSAY_LABEL_BODY_PREFIX, i)
                    };
                    divider(&label);
                    info!("");
                    info!("  {}", enhanced);
                    info!("");
                } else {
                    info!("{}\n", enhanced);
                }
            }
        }
        Err(e) => {
            error!("  ❌ Essay generation failed: {}", e);
        }
    }
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
