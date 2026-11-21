// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # TRAM Assembly Stage
//!
//! Combines the expert-inference conclusion with the KB answer, applies
//! stochastic enhancement and response framing, then appends a URL footer
//! for deep-search results.

#[cfg(feature = "cli")]
use {
    crate::{
        app::tram::rewards::synthesize_evaluative_points,
        cli::{
            display::divider,
            tram::{
                assemble_paragraphs_for, derive_framing_topic,
                response::{build_symbolic_response, is_quality_answer},
                restructure_analogical_response,
            },
        },
        cognition::{
            ReasoningMode,
            code_responder::CodeResponder,
            poem_responder::PoemResponder,
            response_framer::ResponseFramer,
            task_decomposer::{CreativeOutputKind, TaskDecomposer},
        },
        constants::{
            app::{
                CURATED_ONLY_PROBABILITY, DEFAULT_STOCHASTIC_PROBABILITY, MIN_EXPERT_PROSE_WORDS,
            },
            cli_tram_response::MIN_SPARSE_BODY_CHARS,
            code_responder::{
                CODE_IMPROVEMENT_HINT, CODE_INTRO_TEMPLATE, PLACEHOLDER_LANGUAGE,
                PLACEHOLDER_SUBJECT,
            },
            stochastic::{ENABLE_WORDLIST_FALLBACK, PROBABILITY_MAX, PROBABILITY_MIN},
            tram_answer::{ADVANTAGE_KEYWORDS, HISTORICAL_MARKERS, HISTORY_KEYWORDS},
            tram_response::RESPONSE_OPENER_TERMINATOR,
        },
        expert::{inference::InferenceEngine, rule::RuleBase},
        generation::{dedup::GlobalDeduplicator, phrases_mixer::PhrasesMixer, prose::ProseBuilder},
        stochastic::StochasticEnhancer,
        tram::{
            DEFAULT_MIN_RULE_COUNT, query::extract_creative_subject, split_sentences_with_punct,
        },
    },
    tracing::info,
};

#[cfg(all(feature = "cli", feature = "net"))]
use crate::constants::tram_web_fetch::{
    DEEP_SEARCH_URL_FOOTER_BULLET, DEEP_SEARCH_URL_FOOTER_HEADER, URL_REJECT_PATTERNS,
};

/// Splits `text` at the first occurrence of `terminator`, returning
/// `(prefix_including_terminator, remainder)`.
///
/// When `terminator` is not found, returns `("", text)` so callers can
/// detect the no-split case by checking whether the first element is empty.
///
/// # Time Complexity
///
/// O(|text|)
#[cfg(feature = "cli")]
fn split_opener(text: &str, terminator: &str) -> (String, String) {
    if let Some(pos) = text.find(terminator) {
        let end = pos + terminator.len();
        (text[..end].to_string(), text[end..].to_string())
    } else {
        (String::new(), text.to_string())
    }
}

/// Runs the assembly stage.
///
/// 1. Applies expert inference when sufficient rules exist.
/// 2. Runs stochastic enhancement.
/// 3. Frames the response (opener, analogical restructure, creative code block).
/// 4. Appends a URL footer for deep-search runs.
///
/// # Arguments
///
/// * `kb_chunks`           - KB chunk texts for rule extraction.
/// * `combined_answer`     - Best KB answer (may be `None`).
/// * `cold_corpus`         - ThinkLoop cold-memory corpus.
/// * `prompt`              - Original user prompt.
/// * `topic`               - Extracted topic phrase.
/// * `prompt_mode`         - Classified reasoning mode.
/// * `run_seed`            - Per-run seed for phrase variation.
/// * `think`               - Whether the ATE/ThinkLoop was active.
/// * `deep`                - Whether deep-scrape enrichment is active.
/// * `no_frame`            - Skip response framing when `true`.
/// * `verbose`             - Emit diagnostics when `true`.
/// * `probability`         - Optional explicit substitution probability.
/// * `inference_depth`     - Backward-chaining depth limit.
/// * `cf_threshold`        - Minimum certainty factor for expert inference.
/// * `kb_query`            - Derived KB query string.
/// * `all_scraped_urls`    - Scraped URLs from the web-fetch stage.
///
/// # Returns
///
/// The final answer text ready for display.
///
/// # Time Complexity
///
/// O(|rules| × inference_depth + |answer| × stochastic_pass)
#[cfg(feature = "cli")]
#[allow(clippy::too_many_arguments)]
pub fn run_assembly_stage(
    kb_chunks: Vec<String>,
    combined_answer: Option<String>,
    cold_corpus: &str,
    prompt: &str,
    topic: &str,
    prompt_mode: ReasoningMode,
    run_seed: u64,
    think: bool,
    deep: bool,
    no_frame: bool,
    verbose: bool,
    probability: Option<f64>,
    inference_depth: usize,
    cf_threshold: f64,
    kb_query: &str,
    all_scraped_urls: &[String],
) -> String {
    let full_text_for_code: String = combined_answer.as_deref().unwrap_or("").to_string();
    let (answer_text, is_symbolic_fallback) = match combined_answer {
        Some(answer) if is_quality_answer(&answer) => {
            info!("  ✅ KB answer found ({} chars)", answer.len());

            if think && !deep {
                let prompt_lower = prompt.to_ascii_lowercase();
                let prompt_asks_history = HISTORY_KEYWORDS
                    .iter()
                    .any(|&keyword| prompt_lower.contains(keyword));

                let filtered_answer = if !prompt_asks_history {
                    let split_result = split_sentences_with_punct(&answer);
                    let filtered: Vec<String> = split_result
                        .into_iter()
                        .filter(|s| {
                            let lower = s.to_ascii_lowercase();
                            let has_year = lower.chars().any(|c| c.is_ascii_digit());
                            let has_historical_marker = HISTORICAL_MARKERS
                                .iter()
                                .any(|&marker| lower.contains(marker));
                            !(has_year && has_historical_marker)
                        })
                        .collect();
                    if filtered.is_empty() {
                        answer
                    } else {
                        filtered.join(" ")
                    }
                } else {
                    answer
                };

                (
                    assemble_paragraphs_for(&filtered_answer, run_seed, kb_query),
                    false,
                )
            } else if !think && !deep {
                let prompt_lower = prompt.to_ascii_lowercase();
                let prompt_asks_history = HISTORY_KEYWORDS
                    .iter()
                    .any(|&keyword| prompt_lower.contains(keyword));
                let is_advantage_prompt = ADVANTAGE_KEYWORDS
                    .iter()
                    .any(|&kw| prompt_lower.contains(kw));

                let after_history = if !prompt_asks_history {
                    let split_result = split_sentences_with_punct(&answer);
                    let filtered: Vec<String> = split_result
                        .into_iter()
                        .filter(|s| {
                            let lower = s.to_ascii_lowercase();
                            let has_year = lower.chars().any(|c| c.is_ascii_digit());
                            let has_historical_marker = HISTORICAL_MARKERS
                                .iter()
                                .any(|&marker| lower.contains(marker));
                            !(has_year && has_historical_marker)
                        })
                        .collect();
                    if filtered.is_empty() {
                        answer
                    } else {
                        filtered.join(" ")
                    }
                } else {
                    answer
                };

                let final_answer = if is_advantage_prompt {
                    synthesize_evaluative_points(&after_history)
                } else {
                    after_history
                };

                (final_answer, false)
            } else {
                (answer, false)
            }
        }
        Some(answer) if answer.len() >= MIN_SPARSE_BODY_CHARS => {
            info!(
                "  🔀 KB answer too sparse ({} chars). Using as sparse body.",
                answer.len()
            );
            (answer, false)
        }
        Some(_) | None => {
            info!("  🔀 No usable KB answer. Building symbolic response");
            (build_symbolic_response(prompt, cold_corpus), true)
        }
    };

    divider("Expert Inference");
    info!(
        "  ⚙️  Config      : depth={} CF≥{:.2}",
        inference_depth, cf_threshold
    );

    let chunk_refs: Vec<&str> = kb_chunks.iter().map(|s| s.as_str()).collect();
    let rule_base = RuleBase::from_chunks(&chunk_refs);
    let expert_result = if rule_base.rules.len() >= DEFAULT_MIN_RULE_COUNT {
        let engine = InferenceEngine::new(inference_depth, cf_threshold);
        engine.reason(prompt, &rule_base)
    } else {
        None
    };

    let (answer_text, is_symbolic_fallback) = if is_symbolic_fallback {
        (answer_text, true)
    } else if deep {
        (answer_text, false)
    } else if let Some(ref result) = expert_result {
        info!(
            "  🧠 Inference: conclusion={:?} CF={:.2} depth={} rules={}",
            result.conclusion,
            result.certainty,
            result.depth,
            rule_base.rules.len()
        );
        info!("  📋 Explanation: {} steps", result.trace.len());
        if verbose && !result.trace.is_empty() {
            info!("  💬 {}", result.trace.to_prose());
        }
        let builder = ProseBuilder::new(2, 5);
        let expert_prose = builder.build(result, &chunk_refs, run_seed);
        if expert_prose.split_whitespace().count() >= MIN_EXPERT_PROSE_WORDS {
            let deduplicator = GlobalDeduplicator::new(0.65);
            let merged = format!("{}\n\n{}", answer_text, expert_prose);
            (deduplicator.deduplicate_text(&merged), false)
        } else {
            (answer_text, false)
        }
    } else {
        info!(
            "  ℹ  Expert inference: no conclusion (rules={}, threshold={:.2})",
            rule_base.rules.len(),
            cf_threshold
        );
        (answer_text, false)
    };

    divider("Stochastic Enhancement");
    let effective_prob = match probability {
        Some(p) => p.clamp(PROBABILITY_MIN, PROBABILITY_MAX),
        None => DEFAULT_STOCHASTIC_PROBABILITY,
    };
    let needs_mixer_opener =
        !is_symbolic_fallback && !answer_text.is_empty() && !no_frame && !think;
    let (framer_opener, body_before_stochastic) =
        split_opener(&answer_text, RESPONSE_OPENER_TERMINATOR);
    let body_to_enhance = if framer_opener.is_empty() {
        answer_text.clone()
    } else {
        body_before_stochastic
    };
    let enhanced = if is_symbolic_fallback {
        StochasticEnhancer::new(0.0).enhance_detailed(&body_to_enhance)
    } else if ENABLE_WORDLIST_FALLBACK {
        StochasticEnhancer::new(effective_prob).enhance_detailed(&body_to_enhance)
    } else {
        let curated_prob = effective_prob.max(CURATED_ONLY_PROBABILITY);
        StochasticEnhancer::new(curated_prob)
            .curated_only()
            .enhance_detailed(&body_to_enhance)
    };
    let enhanced_full_text = if !framer_opener.is_empty() {
        format!("{}{}", framer_opener, enhanced.text)
    } else if needs_mixer_opener {
        PhrasesMixer::global().apply_opener_if_needed(&enhanced.text, run_seed)
    } else {
        enhanced.text.clone()
    };
    let logged_prob = if is_symbolic_fallback {
        0.0_f64
    } else if !ENABLE_WORDLIST_FALLBACK {
        effective_prob.max(CURATED_ONLY_PROBABILITY)
    } else {
        effective_prob
    };
    info!(
        "  🎲 substitutions={} language={:?} prob={:.2}",
        enhanced.substitutions, enhanced.language, logged_prob
    );

    let final_text = if TaskDecomposer::is_creative_task(prompt) {
        let creative_kind = TaskDecomposer::detect_creative_kind(prompt);
        let kb_corpus: String = kb_chunks.join("\n");
        let subject = extract_creative_subject(prompt);
        if verbose {
            info!("  🎯 Creative subject : '{}'", subject);
            info!("  🎯 Creative kind    : {:?}", creative_kind);
        }
        match creative_kind {
            CreativeOutputKind::Code { ref language } => {
                let extended_corpus = if full_text_for_code.is_empty() {
                    kb_corpus.clone()
                } else {
                    format!("{}\n{}", kb_corpus, full_text_for_code)
                };
                match CodeResponder::generate(language, &subject, &extended_corpus) {
                    Some(code) => {
                        let code_block = CodeResponder::format_code_block(language, &code);
                        let intro = CODE_INTRO_TEMPLATE
                            .replace(PLACEHOLDER_LANGUAGE, language)
                            .replace(PLACEHOLDER_SUBJECT, &subject);
                        format!("{}\n\n{}", intro, code_block)
                    }
                    None => CODE_IMPROVEMENT_HINT
                        .replace(PLACEHOLDER_SUBJECT, &subject)
                        .replace(PLACEHOLDER_LANGUAGE, language),
                }
            }
            CreativeOutputKind::Poem => PoemResponder::compose(&subject, &kb_corpus, run_seed),
            CreativeOutputKind::Story | CreativeOutputKind::Essay | CreativeOutputKind::Generic => {
                assemble_paragraphs_for(&kb_corpus, run_seed, &subject)
            }
        }
    } else if deep || no_frame {
        enhanced_full_text
    } else if think {
        if is_symbolic_fallback {
            enhanced_full_text
        } else {
            let framing_topic = derive_framing_topic(prompt, prompt_mode, topic);
            let framed = ResponseFramer::frame(
                prompt_mode,
                &framing_topic,
                prompt,
                &enhanced.text,
                run_seed,
            );
            if matches!(prompt_mode, ReasoningMode::Analogical) {
                restructure_analogical_response(&framed, topic, run_seed)
            } else {
                framed
            }
        }
    } else {
        enhanced_full_text
    };

    build_url_footer(final_text, deep, all_scraped_urls)
}

/// Appends a URL footer to `text` when deep-search is active and URLs are available.
///
/// Each URL is tested against [`URL_REJECT_PATTERNS`]; only clean URLs are appended.
/// Domain-root URLs (path depth zero or bare `/`) are also rejected to avoid
/// navigation-page links from scraped HTML.
///
/// # Arguments
///
/// * `text`              - The assembled answer text.
/// * `deep`              - Whether deep-search was active.
/// * `all_scraped_urls`  - Candidate URLs from the web-fetch stage.
///
/// # Returns
///
/// `text` with an optional URL footer appended.
#[cfg(all(feature = "cli", feature = "net"))]
fn build_url_footer(text: String, deep: bool, all_scraped_urls: &[String]) -> String {
    if deep && !all_scraped_urls.is_empty() {
        let mut out = text;
        out.push_str(DEEP_SEARCH_URL_FOOTER_HEADER);
        for url in all_scraped_urls {
            let lower = url.to_ascii_lowercase();
            let is_pattern_junk = URL_REJECT_PATTERNS.iter().any(|&pat| lower.contains(pat));
            let is_root_url = url
                .split_once("://")
                .map(|(_, rest)| {
                    let path_start = rest.find('/').map(|i| &rest[i..]).unwrap_or("/");
                    path_start == "/" || path_start.is_empty()
                })
                .unwrap_or(false);
            if !is_pattern_junk && !is_root_url {
                out.push_str(DEEP_SEARCH_URL_FOOTER_BULLET);
                out.push_str(url);
                out.push('\n');
            }
        }
        return out;
    }
    text
}

#[cfg(all(feature = "cli", not(feature = "net")))]
fn build_url_footer(text: String, _deep: bool, _all_scraped_urls: &[String]) -> String {
    text
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
