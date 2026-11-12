// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Stochastic Text Enhancement
//!
//! This module provides [`StochasticEnhancer`] and [`SynonymBank`], which together
//! implement controllable, multi-language word-level variation for generated text.
//!
//! ## Multi-language Support
//!
//! Five synonym tables are compiled into the binary at build time:
//!
//! | Language | Source                        | Entries |
//! |----------|-------------------------------|---------|
//! | English  | Moby Thesaurus II (Grady Ward) | ~25 k   |
//! | Arabic   | Curated modern-standard Arabic | ~22     |
//! | French   | Curated French vocabulary      | ~27     |
//! | German   | Curated German vocabulary      | ~32     |
//! | Spanish  | Curated Spanish vocabulary     | ~33     |
//!
//! The active table is selected automatically via [`detect::detect_language`] or
//! may be overridden by passing a [`detect::LanguageHint`] explicitly.
//!
//! ## Key Optimisation: Compile-time PHF Synonym Tables
//!
//! Every synonym table is a `phf::Map` compiled into the binary at build time.
//! This replaces any runtime `HashMap` construction with **zero-allocation, O(1)**
//! lookups.
//!
//! ## Stop-word Detection
//!
//! English stop words are stored as a compile-time [`phf::Set`], replacing any
//! O(n) linear scan.
//!
//! ## Enhancement Pipeline
//!
//! For each non-stop token in the input text, [`StochasticEnhancer`] samples a
//! uniform Bernoulli draw with probability `p` (default 0.5).  On a "hit", the
//! language-specific synonym table is consulted first; if no entry exists, a
//! same-length word from the system wordlist is substituted.  Case style
//! (ALL_CAPS, Capitalised, lowercase) is mirror-copied to the substituted word.

pub mod arabic;
pub mod detect;
pub mod english;
pub mod french;
pub mod german;
pub mod spanish;

use detect::{LanguageHint, detect_language, synonyms_for};
use phf::{Set, phf_set};
use rand::{Rng, RngExt};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;

/// Default synonym-substitution probability (3%).
const DEFAULT_REPLACEMENT_PROBABILITY: f64 = 0.03;

/// Minimum word length to include from the system dictionary.
#[cfg(not(target_arch = "wasm32"))]
const WORDLIST_MIN_WORD_LEN: usize = 5;

/// Maximum word length to include from the system dictionary.
#[cfg(not(target_arch = "wasm32"))]
const WORDLIST_MAX_WORD_LEN: usize = 14;

/// System dictionary file paths tried in order.
#[cfg(not(target_arch = "wasm32"))]
static SYSTEM_DICT_PATHS: &[&str] = &[
    "/usr/share/dict/american-english",
    "/usr/share/dict/english",
    "/usr/share/dict/words",
    "/usr/dict/words",
];

/// A compile-time perfect-hash set of words that must **never** be substituted.
///
/// These are critical technical terms, programming language names, and common
/// words that would lose meaning or produce nonsense if replaced with synonyms
/// from the Moby Thesaurus II (which contains many archaic/poetic entries).
static PROTECTED_WORDS: Set<&'static str> = phf_set! {
    "rust", "python", "java", "javascript", "typescript", "golang", "swift",
    "kotlin", "scala", "haskell", "ocaml", "erlang", "elixir", "clojure",
    "lisp", "scheme", "prolog", "fortran", "cobol", "pascal", "delphi",
    "assembly", "assembler", "bytecode", "webassembly", "wasm",
    "cargo", "npm", "pip", "gradle",
    "tokio", "rayon", "serde", "clap", "anyhow", "thiserror",
    "mozilla", "graydon", "hoare", "ferris",
    "ownership", "borrow", "borrowing", "lifetime", "lifetimes",
    "trait", "traits", "struct", "enum", "closure",
    "crate", "crates",
    "api", "apis",
    "arc", "rc",
    "option", "ok", "err", "some", "none",
    "println", "eprintln", "unwrap", "expect",
    "map", "filter", "fold", "iter", "into", "collect",
    "push", "pop", "insert", "remove", "get", "set",
    "spawn", "join", "recv", "poll",
    "malloc", "dealloc", "alloc", "allocator", "allocators", "heap", "stack",
    "garbage", "collector", "checker", "dispatcher",
    "github", "gitlab", "bitbucket",
    "docker", "kubernetes", "terraform", "ansible",
    "pytorch", "tensorflow", "keras", "numpy", "pandas", "scipy",
    "linux", "windows", "macos", "android", "ios", "unix", "posix",
    "mathematics", "theory", "system", "complex", "random", "randomness",
    "state", "states", "change", "small", "large", "basic", "principle",
    "deterministic", "nonlinear", "definition", "branch", "study", "scientific",
    "pattern", "patterns", "underlying", "constant", "feedback", "effect",
    "organization", "self", "similarity", "fractal", "fractals",
    "light", "water", "energy", "chemical", "biological", "process", "processes",
    "organism", "organisms", "plant", "plants", "sunlight", "oxygen", "byproduct",
    "autotrophic", "photosynthesis", "photosynthetic", "convert", "conversion",
    "splitting", "produce", "produces", "perform", "performs", "fuel", "metabolism",
    "machine", "learning", "artificial", "intelligence", "algorithm", "algorithms",
    "statistical", "optimization", "optimisation", "foundation", "foundations",
    "theoretical", "perspective", "approximately", "correct", "framework",
    "advances", "advance", "neural", "network", "networks", "surpass", "performance",
    "such", "associate", "fucus", "fungi", "algae", "cyanobacteria", "mostly", "usually",
    "quantum", "computing", "qubit", "qubits", "bit", "bits", "classical", "unit",
    "information", "exponentially", "faster", "hardware", "implementations", "experimental",
    "designed", "specialized", "exist", "linear", "combination", "superposition",
    "working", "created", "developer", "sponsored", "published", "influenced",
    "functional", "immutability", "higher-order", "algebraic", "matching",
    "object-oriented", "traditional", "classes", "inheritance", "general-purpose",
    "february", "may", "release", "anterior", "knowledge", "base", "capillament",
    "type", "safety", "memory", "next", "specific", "subject", "gibberish",
    "january", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "year", "month", "date", "time",
    "because", "context", "web", "query", "returned", "results", "does", "not", "yet", "contain",
    "content", "try", "rephrasing", "your", "adding", "more",
};

/// A compile-time perfect-hash set of English stop words.
///
/// Membership is tested in O(1) via a generated perfect hash, no heap allocation.
static STOP_WORDS: Set<&'static str> = phf_set! {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "its", "it",
    "this", "that", "these", "those", "not", "no", "nor", "so", "yet", "both", "also",
    "as", "if", "than", "then", "from", "into", "through", "during", "before", "after",
    "above", "below", "between", "each", "every", "any", "all"
};

/// Loads the system word list, grouped by word length, for same-length substitution.
///
/// Falls back to an empty map on WASM targets or when no dictionary file is found.
///
/// # Time Complexity
///
/// O(|dict|): linear over the dictionary file size.
///
/// # Space Complexity
///
/// O(|dict|): stores all filtered words in buckets.
#[cfg(not(target_arch = "wasm32"))]
fn load_wordlist_by_length() -> HashMap<usize, Vec<String>> {
    let mut by_length: HashMap<usize, Vec<String>> = HashMap::new();
    for path in SYSTEM_DICT_PATHS {
        if let Ok(content) = fs::read_to_string(path) {
            for word in content.lines() {
                let w = word.trim().to_lowercase();
                if w.len() >= WORDLIST_MIN_WORD_LEN
                    && w.len() <= WORDLIST_MAX_WORD_LEN
                    && w.chars().all(|c| c.is_ascii_alphabetic())
                {
                    by_length.entry(w.len()).or_default().push(w);
                }
            }
            break;
        }
    }
    by_length
}

#[cfg(target_arch = "wasm32")]
fn load_wordlist_by_length() -> HashMap<usize, Vec<String>> {
    HashMap::new()
}

/// Statistics and metadata returned by [`StochasticEnhancer::enhance_detailed`].
///
/// Carries the enhanced text together with diagnostic counters and the
/// detected (or overridden) language so callers can log or assert on them.
#[derive(Debug, Clone)]
pub struct EnhanceOutput {
    /// The enhanced text after synonym substitution.
    pub text: String,

    /// Number of words successfully substituted.
    pub substitutions: usize,

    /// The language detected (or forced) for this enhancement pass.
    pub language: LanguageHint,
}

/// A two-tier synonym lookup bank with multi-language support.
///
/// - **Tier 1**: Language-specific compile-time PHF map, O(1), zero allocation.
/// - **Tier 2**: Runtime system wordlist grouped by word length, O(1) bucket, O(k) sample.
///
/// # Examples
///
/// ```
/// use lmm::stochastic::SynonymBank;
///
/// let bank = SynonymBank::new();
/// assert!(bank.curated_count() > 0);
/// ```
pub struct SynonymBank {
    by_length: HashMap<usize, Vec<String>>,
}

impl SynonymBank {
    /// Creates a new [`SynonymBank`], loading the system wordlist at construction time.
    ///
    /// The curated synonym tables come from compile-time PHF maps (no runtime build cost).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::SynonymBank;
    /// let bank = SynonymBank::new();
    /// assert!(bank.curated_count() > 50);
    /// ```
    pub fn new() -> Self {
        Self {
            by_length: load_wordlist_by_length(),
        }
    }

    /// Samples a random synonym for `word` using the specified language table.
    ///
    /// Tier 1 (language PHF map) is tried first; if the word is not in the curated
    /// table, Tier 2 (same-length wordlist) is used.
    ///
    /// # Arguments
    ///
    /// * `word`: The word to replace (case-insensitive lookup).
    /// * `rng`: A mutable random number generator.
    /// * `hint`: The language table to consult.
    ///
    /// # Returns
    ///
    /// `Some(String)` containing a synonym, or `None` when no alternative is available.
    ///
    /// # Time Complexity
    ///
    /// O(1) amortised for the PHF lookup; O(k) for sampling from the wordlist bucket.
    ///
    /// # Space Complexity
    ///
    /// O(1) for the lookup itself; the returned `String` is O(|synonym|).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::{SynonymBank, detect::LanguageHint};
    ///
    /// let bank = SynonymBank::new();
    /// let mut rng = rand::rng();
    /// let syn = bank.candidate("chaos", &mut rng, LanguageHint::English);
    /// assert!(syn.is_some());
    /// ```
    pub fn candidate<R: Rng>(&self, word: &str, rng: &mut R, hint: LanguageHint) -> Option<String> {
        let lower = word.to_lowercase();
        let table = synonyms_for(hint);

        if let Some(syns) = table.get(lower.as_str())
            && !syns.is_empty()
        {
            let idx = rng.random_range(0..syns.len());
            return Some(syns[idx].to_string());
        }

        if let Some(bucket) = self.by_length.get(&lower.len()) {
            let candidates: Vec<&String> = bucket.iter().filter(|w| w.as_str() != lower).collect();
            if !candidates.is_empty() {
                let idx = rng.random_range(0..candidates.len());
                return Some(candidates[idx].clone());
            }
        }
        None
    }

    /// Samples a curated synonym for `word` **without** the system-wordlist fallback.
    ///
    /// Unlike [`candidate`](Self::candidate), this method returns `None` when the
    /// word is absent from the language-specific PHF table, ensuring that generated
    /// or domain-specific vocabulary is never replaced with an arbitrary dictionary
    /// word of the same length.
    ///
    /// # Arguments
    ///
    /// * `word`: The word to replace (case-insensitive lookup).
    /// * `rng` : A mutable random number generator.
    /// * `hint`: The language table to consult.
    ///
    /// # Returns
    ///
    /// `Some(String)` when a curated synonym exists, `None` otherwise.
    ///
    /// # Time Complexity
    ///
    /// O(1): single PHF map lookup.
    ///
    /// # Space Complexity
    ///
    /// O(1) for the lookup; O(|synonym|) for the returned value.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::{SynonymBank, detect::LanguageHint};
    ///
    /// let bank = SynonymBank::new();
    /// let mut rng = rand::rng();
    /// let syn = bank.curated_candidate("chaos", &mut rng, LanguageHint::English);
    /// assert!(syn.is_some());
    /// let none = bank.curated_candidate("zzzzzzzzz", &mut rng, LanguageHint::English);
    /// assert!(none.is_none());
    /// ```
    pub fn curated_candidate<R: Rng>(
        &self,
        word: &str,
        rng: &mut R,
        hint: LanguageHint,
    ) -> Option<String> {
        let lower = word.to_lowercase();
        let table = synonyms_for(hint);
        if let Some(syns) = table.get(lower.as_str())
            && !syns.is_empty()
        {
            let idx = rng.random_range(0..syns.len());
            return Some(syns[idx].to_string());
        }
        None
    }

    /// Returns the number of entries in the English curated synonym table.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::SynonymBank;
    /// assert!(SynonymBank::new().curated_count() > 50);
    /// ```
    pub fn curated_count(&self) -> usize {
        english::CURATED_SYNONYMS.len()
    }

    /// Returns the total number of words in the loaded system wordlist.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::SynonymBank;
    /// let _ = SynonymBank::new().wordlist_len();
    /// ```
    pub fn wordlist_len(&self) -> usize {
        self.by_length.values().map(|v| v.len()).sum()
    }
}

impl Default for SynonymBank {
    fn default() -> Self {
        Self::new()
    }
}

/// A probabilistic, multi-language word-substitution engine for stochastic text variation.
///
/// At each token, a Bernoulli trial with probability `p` determines whether to
/// substitute the word.  Stop words (from the compile-time `STOP_WORDS` PHF set)
/// are always preserved.  Case style (ALL_CAPS, Capitalised, lowercase) is
/// mirror-copied to the substituted word.
///
/// When `curated_only` is `true`, the random system-wordlist fallback is disabled:
/// a word is only replaced when a **curated synonym** exists in the language table.
/// This prevents technical or generated vocabulary (e.g. "structural", "recursive")
/// from being replaced with unrelated dictionary words.
///
/// # Examples
///
/// ```
/// use lmm::stochastic::StochasticEnhancer;
///
/// let enhancer = StochasticEnhancer::with_default_probability();
/// let output = enhancer.enhance("chaos governs the universe");
/// assert!(!output.is_empty());
/// ```
pub struct StochasticEnhancer {
    bank: SynonymBank,
    probability: f64,
    /// When `true`, only substitutes words that have a curated synonym entry.
    curated_only: bool,
}

impl StochasticEnhancer {
    /// Creates a [`StochasticEnhancer`] with a custom substitution probability.
    ///
    /// `probability` is clamped to `[0.0, 1.0]`.
    ///
    /// # Arguments
    ///
    /// * `probability`: Per-word substitution probability.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    /// let e = StochasticEnhancer::new(0.3);
    /// assert_eq!(e.probability(), 0.3);
    /// ```
    pub fn new(probability: f64) -> Self {
        Self {
            bank: SynonymBank::new(),
            probability: probability.clamp(0.0, 1.0),
            curated_only: false,
        }
    }

    /// Creates a [`StochasticEnhancer`] with `p = 0.5`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    /// assert_eq!(StochasticEnhancer::with_default_probability().probability(), 0.5);
    /// ```
    pub fn with_default_probability() -> Self {
        Self::new(DEFAULT_REPLACEMENT_PROBABILITY)
    }

    /// Enables curated-only mode: only substitute words that have a curated synonym entry.
    ///
    /// When enabled, the random system-wordlist fallback is disabled so technical
    /// or generated vocabulary (e.g. "structural", "recursive") is never replaced
    /// with an unrelated dictionary word of the same length.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    /// let e = StochasticEnhancer::new(0.4).curated_only();
    /// assert!(e.is_curated_only());
    /// ```
    pub fn curated_only(mut self) -> Self {
        self.curated_only = true;
        self
    }

    /// Returns `true` when curated-only mode is active.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    /// assert!(!StochasticEnhancer::new(0.3).is_curated_only());
    /// assert!(StochasticEnhancer::new(0.3).curated_only().is_curated_only());
    /// ```
    pub fn is_curated_only(&self) -> bool {
        self.curated_only
    }

    /// Returns the configured substitution probability.
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Enhances all lines of `text` by stochastic synonym substitution.
    ///
    /// The language is auto-detected via [`detect_language`]; use
    /// [`enhance_with_hint`](Self::enhance_with_hint) to override.
    ///
    /// # Arguments
    ///
    /// * `text`: Input text (may contain newlines).
    ///
    /// # Returns
    ///
    /// Enhanced `String` with the same line structure as `text`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    ///
    /// let e = StochasticEnhancer::new(1.0);
    /// let out = e.enhance("chaos\nenergy");
    /// assert_eq!(out.lines().count(), 2);
    /// ```
    pub fn enhance(&self, text: &str) -> String {
        let hint = detect_language(text);
        self.enhance_with_hint(text, hint)
    }

    /// Enhances `text` using a caller-supplied [`LanguageHint`].
    ///
    /// Identical to [`enhance`](Self::enhance) but skips language detection.
    ///
    /// # Arguments
    ///
    /// * `text`: Input text (may contain newlines).
    /// * `hint`: The language table to use.
    ///
    /// # Time Complexity
    ///
    /// O(|text|): processes every token once.
    ///
    /// # Space Complexity
    ///
    /// O(|text|): builds the output string.
    pub fn enhance_with_hint(&self, text: &str, hint: LanguageHint) -> String {
        let mut rng = rand::rng();
        text.split('\n')
            .map(|line| self.enhance_line(line, &mut rng, hint))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Enhances `text` and returns an [`EnhanceOutput`] with diagnostic metadata.
    ///
    /// The language is auto-detected.  The returned [`EnhanceOutput`] carries the
    /// enhanced text, the substitution count, and the detected language.
    ///
    /// # Arguments
    ///
    /// * `text`: Input text (may contain newlines).
    ///
    /// # Returns
    ///
    /// An [`EnhanceOutput`] struct.
    ///
    /// # Time Complexity
    ///
    /// O(|text|): processes every token once.
    ///
    /// # Space Complexity
    ///
    /// O(|text|): builds the output string.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    ///
    /// let e = StochasticEnhancer::new(1.0);
    /// let out = e.enhance_detailed("mathematics governs the cosmos");
    /// assert!(out.substitutions > 0 || out.text.len() > 0);
    /// ```
    pub fn enhance_detailed(&self, text: &str) -> EnhanceOutput {
        let language = detect_language(text);
        let mut rng = rand::rng();
        let mut substitutions: usize = 0;
        let enhanced = text
            .split('\n')
            .map(|line| self.enhance_line_counted(line, &mut rng, language, &mut substitutions))
            .collect::<Vec<_>>()
            .join("\n");
        EnhanceOutput {
            text: enhanced,
            substitutions,
            language,
        }
    }

    /// Enhances a single line, counting substitutions into `counter`.
    fn enhance_line_counted<R: Rng>(
        &self,
        line: &str,
        rng: &mut R,
        hint: LanguageHint,
        counter: &mut usize,
    ) -> String {
        let mut result: Vec<String> = Vec::new();
        let mut is_first = true;

        for raw in line.split_whitespace() {
            let (prefix, core, suffix) = split_token(raw);
            let lower = core.to_lowercase();
            let is_stop = STOP_WORDS.contains(lower.as_str());
            let was_capitalized = core
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);
            let was_all_caps = !core.is_empty() && core.chars().all(|c| c.is_uppercase());
            let is_proper_noun = was_capitalized && !is_first;

            let replacement = if !is_stop
                && !is_proper_noun
                && !PROTECTED_WORDS.contains(lower.as_str())
                && !core.is_empty()
                && !core.chars().any(|c| c.is_ascii_digit())
                && core.chars().all(|c| c.is_alphabetic() || c == '-')
                && rng.random_bool(self.probability)
            {
                let candidate = if self.curated_only {
                    self.bank.curated_candidate(&lower, rng, hint)
                } else {
                    self.bank.candidate(&lower, rng, hint)
                };
                match candidate {
                    Some(syn) => {
                        *counter += 1;
                        if was_all_caps {
                            syn.to_uppercase()
                        } else if was_capitalized || is_first {
                            capitalize(&syn)
                        } else {
                            syn
                        }
                    }
                    None => core.to_string(),
                }
            } else {
                core.to_string()
            };

            result.push(format!("{}{}{}", prefix, replacement, suffix));
            is_first = false;
        }

        result.join(" ")
    }

    /// Enhances a single line without counting substitutions.
    fn enhance_line<R: Rng>(&self, line: &str, rng: &mut R, hint: LanguageHint) -> String {
        let mut counter: usize = 0;
        self.enhance_line_counted(line, rng, hint, &mut counter)
    }
}

/// Splits a raw token `"(word,"` into `("(", "word", ",")`.
///
/// Strips leading punctuation/symbols into `prefix` and trailing punctuation into
/// `suffix`.
///
/// # Arguments
///
/// * `raw`: A whitespace-separated token.
///
/// # Returns
///
/// `(&str, &str, &str)`: `(prefix, core, suffix)` where `core` is the alphabetic run.
///
/// # Time Complexity
///
/// O(|raw|).
///
/// # Space Complexity
///
/// O(1): returns sub-slices of `raw`.
fn split_token(raw: &str) -> (&str, &str, &str) {
    let prefix_end = raw.find(|c: char| c.is_alphabetic()).unwrap_or(raw.len());
    let content_start = prefix_end;
    if content_start >= raw.len() {
        return ("", raw, "");
    }
    let suffix_start = raw[content_start..]
        .rfind(|c: char| c.is_alphabetic())
        .map(|i| {
            content_start
                + i
                + raw[content_start + i..]
                    .chars()
                    .next()
                    .map(|ch| ch.len_utf8())
                    .unwrap_or(1)
        })
        .unwrap_or(raw.len());
    (
        &raw[..content_start],
        &raw[content_start..suffix_start],
        &raw[suffix_start..],
    )
}

/// Uppercases the first character and leaves the rest unchanged.
///
/// # Arguments
///
/// * `s`: The input string to capitalise.
///
/// # Returns
///
/// A new `String` with only the first character uppercased.
///
/// # Examples
///
/// ```
/// use lmm::stochastic::capitalize;
///
/// assert_eq!(capitalize("hello"), "Hello");
/// assert_eq!(capitalize(""), "");
/// ```
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Returns `true` when `word` is an English stop word according to the compile-time PHF set.
///
/// The lookup is O(1) with no heap allocation.
///
/// # Arguments
///
/// * `word`: A lowercase word (case-sensitive lookup against the PHF set).
///
/// # Examples
///
/// ```
/// use lmm::stochastic::is_stop_word;
///
/// assert!(is_stop_word("the"));
/// assert!(!is_stop_word("chaos"));
/// ```
pub fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.contains(word)
}
