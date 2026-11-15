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

/// Handles the `predict` subcommand, symbolic text continuation.
///
/// # Arguments
///
/// * `input`          - File path to load as seed corpus.
/// * `text`           - Fallback inline seed text.
/// * `window`         - N-gram window width.
/// * `predict_length` - Number of words to predict.
/// * `iterations`     - GP iterations.
/// * `depth`          - Recursion depth.
/// * `dictionary`     - Optional path to a word lexicon.
/// * `stochastic`     - Whether to apply synonym substitution.
/// * `probability`    - Substitution probability (used when `stochastic` is `true`).
/// * `verbose`        - Verbose output flag.
#[cfg(feature = "cli")]
pub fn handle_predict(
    input: String,
    text: String,
    window: usize,
    predict_length: usize,
    iterations: usize,
    depth: usize,
    dictionary: Option<String>,
    stochastic: bool,
    probability: f64,
    verbose: bool,
) -> anyhow::Result<()> {
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
        println!("{}{}", source, cont);
    }
    Ok(())
}

/// Handles the `summarize` subcommand, extractive key-sentence summarisation.
///
/// # Arguments
///
/// * `input`      - File path to load as source corpus.
/// * `text`       - Fallback inline corpus text.
/// * `sentences`  - Number of sentences to extract.
/// * `iterations` - GP iterations.
/// * `depth`      - Recursion depth.
/// * `stochastic` - Whether to apply synonym substitution.
/// * `probability`- Substitution probability.
/// * `verbose`    - Verbose output flag.
#[cfg(feature = "cli")]
pub fn handle_summarize(
    input: String,
    text: String,
    sentences: usize,
    iterations: usize,
    depth: usize,
    stochastic: bool,
    probability: f64,
    verbose: bool,
) -> anyhow::Result<()> {
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
                    println!("{}", enhanced);
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
                println!("{}", enhanced);
            }
        }
        Err(e) => {
            error!("  ❌ Generation failed: {}", e);
        }
    }
    Ok(())
}

/// Handles the `paragraph` subcommand, multi-sentence symbolic paragraph generation.
///
/// # Arguments
///
/// * `input`      - File path to load as seed.
/// * `text`       - Fallback inline seed text.
/// * `sentences`  - Number of sentences in the paragraph.
/// * `iterations` - GP iterations.
/// * `depth`      - Recursion depth.
/// * `stochastic` - Whether to apply synonym substitution.
/// * `probability`- Substitution probability.
/// * `verbose`    - Verbose output flag.
#[cfg(feature = "cli")]
pub fn handle_paragraph(
    input: String,
    text: String,
    sentences: usize,
    iterations: usize,
    depth: usize,
    stochastic: bool,
    probability: f64,
    verbose: bool,
) -> anyhow::Result<()> {
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
                println!("{}", enhanced);
            }
        }
        Err(e) => {
            error!("  ❌ Generation failed: {}", e);
        }
    }
    Ok(())
}

/// Handles the `essay` subcommand, multi-paragraph symbolic essay generation.
///
/// # Arguments
///
/// * `input`      - File path to load as topic seed.
/// * `text`       - Fallback inline topic text.
/// * `paragraphs` - Number of paragraphs to generate.
/// * `sentences`  - Number of sentences per paragraph.
/// * `iterations` - GP iterations.
/// * `depth`      - Recursion depth.
/// * `stochastic` - Whether to apply synonym substitution.
/// * `probability`- Substitution probability.
/// * `verbose`    - Verbose output flag.
#[cfg(feature = "cli")]
pub fn handle_essay(
    input: String,
    text: String,
    paragraphs: usize,
    sentences: usize,
    iterations: usize,
    depth: usize,
    stochastic: bool,
    probability: f64,
    verbose: bool,
) -> anyhow::Result<()> {
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
                println!("{}\n", essay.title);
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
                    println!("{}\n", enhanced);
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

