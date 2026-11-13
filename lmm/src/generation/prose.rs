// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Novel Prose Generation: ProseBuilder
//!
//! Builds **novel, coherent paragraphs** grounded in expert inference results
//! and raw KB knowledge chunks.
//!
//! Unlike the stochastic synonym-substitution layer (which modifies existing
//! sentences), `ProseBuilder` **constructs new sentences** by filling typed
//! template slots with tokens derived from the [`InferenceResult`] proof tree
//! and top-K KB chunks.
//!
//! ## Generation pipeline
//!
//! ```text
//! InferenceResult.trace  -> ProofStep conclusions (derived propositions)
//! KB chunks              -> Top extractive sentences (factual grounding)
//! Template bank          -> Structural scaffolding (slot: subject, predicate, object)
//! run_seed               -> Template selection, sentence count, paragraph count
//!      ↓
//! CoherenceScorer::reorder   (maximise topical flow)
//!      ↓
//! assemble_paragraphs        (discourse connectors between paragraphs)
//!      ↓
//! Final novel prose
//! ```
//!
//! Every call with the same knowledge but a different `run_seed` produces
//! structurally different output because the template choice, sentence count,
//! and paragraph layout are all seeded from `SystemTime`.

use crate::expert::inference::InferenceResult;
use crate::generation::coherence::CoherenceScorer;
use crate::generation::filter::SentenceFilter;
use std::collections::HashSet;

/// Paragraph templates.  `{subject}`, `{predicate}`, `{object}`, and
/// `{conclusion}` are replaced with tokens from the inference result.
///
/// All templates that inline `{conclusion}` directly (as a raw noun phrase)
/// have been removed to prevent ungrammatical output such as
/// _"Goal Webassembly is facilitate high-performance applications, which..."_.
/// Instead, `{subject}`, `{predicate}`, and `{object}` are used in
/// structured clauses that read naturally regardless of the raw extraction.
static TEMPLATES: &[&str] = &[
    "In practice, {subject} {predicate} {object}.",
    "More precisely, {subject} {predicate} {object}.",
    "This is because {subject} {predicate} {object}.",
    "To be specific, {subject} {predicate} {object}.",
    "It is well established that {subject} {predicate} {object}.",
    "At its core, {subject} {predicate} {object}.",
    "Notably, {subject} {predicate} {object}.",
    "Put simply, {subject} {predicate} {object}.",
    "Broadly speaking, {subject} {predicate} {object}.",
    "In technical terms, {subject} {predicate} {object}.",
    "By design, {subject} {predicate} {object}.",
    "In essence, {subject} {predicate} {object}.",
    "From a practical standpoint, {subject} {predicate} {object}.",
    "To elaborate, {subject} {predicate} {object}.",
    "On the technical side, {subject} {predicate} {object}.",
    "In this context, {subject} {predicate} {object}.",
    "Fundamentally, {subject} {predicate} {object}.",
    "Specifically, {subject} {predicate} {object}.",
    "Consequently, {subject} {predicate} {object}.",
    "As a result, {subject} {predicate} {object}.",
    "Therefore, {subject} {predicate} {object}.",
    "Thus, {subject} {predicate} {object}.",
    "Hence, {subject} {predicate} {object}.",
    "Moreover, {subject} {predicate} {object}.",
    "Furthermore, {subject} {predicate} {object}.",
    "In addition, {subject} {predicate} {object}.",
    "Significantly, {subject} {predicate} {object}.",
    "Importantly, {subject} {predicate} {object}.",
    "Crucially, {subject} {predicate} {object}.",
    "Essentially, {subject} {predicate} {object}.",
    "Ultimately, {subject} {predicate} {object}.",
    "Principally, {subject} {predicate} {object}.",
    "Primarily, {subject} {predicate} {object}.",
    "Particularly, {subject} {predicate} {object}.",
    "Especially, {subject} {predicate} {object}.",
];

/// Discourse connector phrases prepended to non-first paragraphs.
static CONNECTORS: &[&str] = &[
    "Furthermore,",
    "In addition,",
    "Notably,",
    "Moreover,",
    "As a result,",
    "In practice,",
    "Interestingly,",
    "Consequently,",
    "Building on this,",
    "To elaborate,",
    "More specifically,",
    "On the technical side,",
];

/// Builds novel, topic-grounded paragraphs from inference results and KB chunks.
///
/// The number of sentences (3-6) and paragraphs (1-3) varies per `run_seed`,
/// ensuring different output on every invocation.
///
/// # Examples
///
/// ```rust
/// use lmm::generation::prose::ProseBuilder;
/// use lmm::expert::inference::InferenceResult;
/// use lmm::expert::explain::ExplanationTrace;
///
/// let result = InferenceResult {
///     conclusion: "rust is memory-safe".to_string(),
///     certainty: 0.8,
///     trace: ExplanationTrace::new(),
///     depth: 1,
/// };
/// let chunks = ["Rust uses an ownership model to manage memory without a garbage collector."];
/// let builder = ProseBuilder::new(2, 4);
/// let prose = builder.build(&result, &chunks, 12345);
/// assert!(!prose.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct ProseBuilder {
    /// Target number of paragraphs (actual count is seeded ∈ [1, target_paragraphs]).
    pub target_paragraphs: usize,
    /// Target sentences per paragraph (actual count is seeded ∈ [2, target_sentences]).
    pub target_sentences: usize,
}

impl ProseBuilder {
    /// Creates a new `ProseBuilder`.
    ///
    /// # Arguments
    ///
    /// * `target_paragraphs` - Maximum paragraph count (1-3 is typical).
    /// * `target_sentences`  - Maximum sentences per paragraph (2-6 is typical).
    ///
    /// # Time Complexity
    ///
    /// O(1)
    pub fn new(target_paragraphs: usize, target_sentences: usize) -> Self {
        Self {
            target_paragraphs: target_paragraphs.max(1),
            target_sentences: target_sentences.max(2),
        }
    }

    /// Builds novel prose from an inference result and KB chunks.
    ///
    /// Templates are sampled without replacement within a single `build` call,
    /// so the same boilerplate sentence structure never appears twice in one
    /// output.  Near-duplicate sentences (identical conclusions expressed in
    /// different templates) are also filtered before assembly.
    ///
    /// # Arguments
    ///
    /// * `result`     - The inference result with proof trace and CF.
    /// * `kb_chunks`  - Raw text strings from `KnowledgeIndex::query`.
    /// * `run_seed`   - Per-run nanosecond entropy for randomisation.
    ///
    /// # Time Complexity
    ///
    /// O(|kb_chunks| + |trace| + sentence_count²)
    ///
    /// # Space Complexity
    ///
    /// O(|output|)
    pub fn build(&self, result: &InferenceResult, kb_chunks: &[&str], run_seed: u64) -> String {
        let mut seed = run_seed;

        let sentence_count = 2
            + (lcg_next(&mut seed) % (self.target_sentences.saturating_sub(1) as u64 + 1)) as usize;
        let para_count = 1 + (lcg_next(&mut seed) % (self.target_paragraphs as u64)) as usize;

        let mut all_sentences: Vec<String> = Vec::new();
        let mut used_templates: HashSet<usize> = HashSet::new();
        let mut seen_conclusions: HashSet<String> = HashSet::new();

        let (subj, pred, obj) = extract_spo(&result.conclusion);

        for step in &result.trace.steps {
            if step.conclusion.is_empty() {
                continue;
            }
            let conc_key = step
                .conclusion
                .split_whitespace()
                .take(5)
                .collect::<Vec<_>>()
                .join(" ")
                .to_ascii_lowercase();
            if seen_conclusions.contains(&conc_key) {
                continue;
            }
            seen_conclusions.insert(conc_key);
            let tmpl_idx = pick_template_unique(&mut seed, &used_templates);
            used_templates.insert(tmpl_idx);
            let (step_subj, step_pred, step_obj) = extract_spo(&step.conclusion);
            let sentence = fill_template(
                TEMPLATES[tmpl_idx],
                &step_subj,
                &step_pred,
                &step_obj,
                &step.conclusion,
            );
            let capped = capitalise(sentence);
            if !capped.is_empty() {
                all_sentences.push(capped);
            }
        }

        let conc_key = result
            .conclusion
            .split_whitespace()
            .take(5)
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();
        if !seen_conclusions.contains(&conc_key) {
            let tmpl_idx = pick_template_unique(&mut seed, &used_templates);
            used_templates.insert(tmpl_idx);
            let conclusion_sentence =
                fill_template(TEMPLATES[tmpl_idx], &subj, &pred, &obj, &result.conclusion);
            let capped = capitalise(conclusion_sentence);
            if !capped.is_empty() {
                all_sentences.push(capped);
            }
        }

        for chunk in kb_chunks.iter().take(4) {
            for raw_sent in chunk.split(['.', '!', '?']) {
                let trimmed = raw_sent.trim();
                if trimmed.split_whitespace().count() >= 4 {
                    let sent = if trimmed.ends_with(['.', '!', '?']) {
                        trimmed.to_string()
                    } else {
                        format!("{trimmed}.")
                    };
                    if !SentenceFilter::is_metadata_line(&sent)
                        && !SentenceFilter::is_header_fragment(&sent)
                        && !SentenceFilter::is_disambiguation_fragment(&sent)
                    {
                        all_sentences.push(capitalise(sent));
                    }
                }
            }
        }

        all_sentences.dedup();

        let scorer = CoherenceScorer;
        let ordered = scorer.reorder(&all_sentences);

        if ordered.is_empty() {
            return String::new();
        }

        let total = ordered.len().min(sentence_count * para_count).max(1);
        let sentences = &ordered[..total];

        let per_para = (sentences.len() / para_count).max(1);
        let mut paragraphs: Vec<String> = Vec::new();
        let mut offset = 0usize;

        for p in 0..para_count {
            let take = if p == para_count - 1 {
                sentences.len().saturating_sub(offset)
            } else {
                per_para
            };
            if take == 0 {
                break;
            }
            let end = (offset + take).min(sentences.len());
            let chunk = &sentences[offset..end];
            offset = end;

            let para = if p > 0 && !chunk.is_empty() {
                let conn = CONNECTORS[(seed as usize + p) % CONNECTORS.len()];
                let first = &chunk[0];
                let lowered = lower_first(first);
                let joined_first = format!("{conn} {lowered}");
                let mut parts = vec![joined_first];
                for s in &chunk[1..] {
                    parts.push(s.clone());
                }
                parts.join(" ")
            } else {
                chunk.join(" ")
            };

            if !para.trim().is_empty() {
                paragraphs.push(capitalise(para));
            }
        }

        paragraphs.join("\n\n")
    }
}

/// Extracts a (subject, predicate, object) triple from a simple conclusion string.
///
/// Falls back to ("it", "is", conclusion) when no relation verb is found.
///
/// # Time Complexity
///
/// O(|conclusion|)
fn extract_spo(conclusion: &str) -> (String, String, String) {
    static REL_VERBS: &[&str] = &[
        "is",
        "are",
        "enables",
        "provides",
        "requires",
        "supports",
        "uses",
        "prevents",
        "guarantees",
        "has",
        "ensures",
        "allows",
    ];
    let words: Vec<&str> = conclusion.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        let lower = word.to_ascii_lowercase();
        let verb = lower.trim_matches(|c: char| !c.is_ascii_alphabetic());
        if REL_VERBS.contains(&verb) && i > 0 {
            let subj = words[..i].join(" ");
            let obj = words[i + 1..].join(" ");
            return (subj, verb.to_string(), obj);
        }
    }
    ("it".to_string(), "is".to_string(), conclusion.to_string())
}

/// Fills a template string by replacing slot markers.
///
/// The `{conclusion}` slot is normalised via [`normalise_conclusion`] before
/// insertion so inference engine output (which is lowercase and may lack
/// articles) reads naturally inside a sentence.
///
/// # Time Complexity
///
/// O(|template|)
fn fill_template(
    template: &str,
    subject: &str,
    predicate: &str,
    object: &str,
    conclusion: &str,
) -> String {
    let norm_conc = normalise_conclusion(conclusion);
    template
        .replace("{subject}", subject)
        .replace("{predicate}", predicate)
        .replace("{object}", object)
        .replace("{conclusion}", &norm_conc)
}

/// Normalises an inference-engine conclusion string for use inside a prose sentence.
///
/// Inference conclusions are produced in lowercase without articles
/// (e.g. `"borrow checker is core component"`).  This function:
///
/// 1. Capitalises every word that looks like a known proper noun or brand name.
/// 2. Does **not** insert articles, the template wording already supplies
///    enough context for the sentence to read naturally.
///
/// # Time Complexity
///
/// O(|conclusion|)
fn normalise_conclusion(conclusion: &str) -> String {
    static PROPER_NOUNS: phf::Set<&'static str> = phf::phf_set! {
        "rust", "python", "java", "javascript", "typescript", "golang", "swift",
        "kotlin", "scala", "haskell", "ocaml", "erlang", "elixir", "clojure",
        "linux", "windows", "macos", "android", "ios", "unix",
        "mozilla", "google", "microsoft", "apple", "amazon", "meta",
        "tokio", "cargo", "wasm", "webassembly",
        "stackoverflow",
    };
    conclusion
        .split_whitespace()
        .map(|word| {
            let lower = word.to_ascii_lowercase();
            let clean: String = lower.chars().filter(|c| c.is_ascii_alphabetic()).collect();
            if PROPER_NOUNS.contains(clean.as_str()) {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(c) => {
                        let up: String = c.to_uppercase().collect();
                        up + chars.as_str()
                    }
                }
            } else {
                word.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Picks a template index not already in `used`, falling back to any index when
/// all templates have been exhausted.
///
/// This prevents the same boilerplate structure from appearing twice within a
/// single `ProseBuilder::build` call.
///
/// # Time Complexity
///
/// O(|TEMPLATES|)
fn pick_template_unique(seed: &mut u64, used: &HashSet<usize>) -> usize {
    let start = (lcg_next(seed) as usize) % TEMPLATES.len();
    for offset in 0..TEMPLATES.len() {
        let idx = (start + offset) % TEMPLATES.len();
        if !used.contains(&idx) {
            return idx;
        }
    }
    start
}

/// Minimal LCG pseudo-random number generator step.
///
/// Uses the multiplier and addend from Donald Knuth's MMIX.
///
/// # Time Complexity
///
/// O(1)
fn lcg_next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

/// Capitalises the first Unicode character of a string.
///
/// # Time Complexity
///
/// O(|s|)
fn capitalise(s: impl Into<String>) -> String {
    let s = s.into();
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let upper: String = c.to_uppercase().collect();
            upper + chars.as_str()
        }
    }
}

/// Lowercases the first Unicode character of a string.
///
/// # Time Complexity
///
/// O(|s|)
fn lower_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let lower: String = c.to_lowercase().collect();
            lower + chars.as_str()
        }
    }
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
