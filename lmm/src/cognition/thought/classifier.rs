// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Input Classifier
//!
//! Classifies a raw user prompt into one of seven [`ReasoningMode`]s using a
//! lightweight, training-free **scored token-matching** approach.
//!
//! Each mode has a set of **signal words** weighted by their diagnostic strength.
//! The input is scanned for these signals; the mode with the highest total score
//! wins.  Ties are broken by the order of definition (highest priority listed
//! first in the enum).
//!
//! This O(|prompt|) classification determines which sub-aspects are explored,
//! what `StepMode` the [`crate::cognition::thought::AdaptiveThoughtEngine`]
//! applies per step, and how the [`crate::cognition::thought::decompose::GoalDecomposer`]
//! decomposes the goal.
//!
//! ## Examples
//!
//! ```rust
//! use lmm::cognition::thought::classifier::{InputClassifier, ReasoningMode};
//!
//! assert_eq!(InputClassifier::classify("what is Rust?"), ReasoningMode::FactualLookup);
//! assert_eq!(InputClassifier::classify("why does Rust avoid GC?"), ReasoningMode::Causal);
//! assert_eq!(InputClassifier::classify("rust"), ReasoningMode::Exploratory);
//! ```
//!
//! ## See Also
//!
//! * [Question classification - Wikipedia](https://en.wikipedia.org/wiki/Question#Semantic_classification)

/// The kind of reasoning a user prompt calls for.
///
/// Each variant drives a distinct reasoning strategy in the
/// [`crate::cognition::thought::AdaptiveThoughtEngine`].
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::classifier::ReasoningMode;
///
/// let mode = ReasoningMode::Causal;
/// assert_eq!(mode.label(), "causal");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningMode {
    /// "What is X?", "Who is X?", "When was X?", retrieve a known fact.
    FactualLookup,
    /// "Why does X?", "What causes X?", "How does X happen?", trace causes.
    Causal,
    /// "X like Y", "X vs Y", "compare X and Y", examine similarities/differences.
    Analogical,
    /// Bare noun, "define X", "explain X", establish meaning and scope.
    Definitional,
    /// "How do I X?", "steps to X", "how to X", produce ordered instructions.
    Procedural,
    /// Single word, creative prompt, ambiguous question, broad exploration.
    Exploratory,
    /// "Is X good?", "which is better X or Y?", weigh evidence and judge.
    Evaluative,
}

impl ReasoningMode {
    /// Returns a short lowercase label for this mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::classifier::ReasoningMode;
    ///
    /// assert_eq!(ReasoningMode::FactualLookup.label(), "factual");
    /// assert_eq!(ReasoningMode::Procedural.label(), "procedural");
    /// ```
    pub fn label(&self) -> &'static str {
        match self {
            Self::FactualLookup => "factual",
            Self::Causal => "causal",
            Self::Analogical => "analogical",
            Self::Definitional => "definitional",
            Self::Procedural => "procedural",
            Self::Exploratory => "exploratory",
            Self::Evaluative => "evaluative",
        }
    }
}

/// Signal entries: `(word, score_delta)`.  Higher scores mean stronger evidence
/// for the associated mode.  Scores are additive across a prompt.
struct ModeSignals {
    factual: &'static [(&'static str, u32)],
    causal: &'static [(&'static str, u32)],
    analogical: &'static [(&'static str, u32)],
    definitional: &'static [(&'static str, u32)],
    procedural: &'static [(&'static str, u32)],
    evaluative: &'static [(&'static str, u32)],
}

static SIGNALS: ModeSignals = ModeSignals {
    factual: &[
        ("what", 3),
        ("who", 5),
        ("when", 5),
        ("where", 5),
        ("which", 3),
        ("is", 1),
        ("are", 1),
        ("was", 2),
        ("were", 2),
        ("name", 2),
        ("list", 2),
        ("tell", 1),
    ],
    causal: &[
        ("why", 8),
        ("cause", 6),
        ("causes", 6),
        ("caused", 6),
        ("because", 5),
        ("reason", 5),
        ("reasons", 5),
        ("effect", 4),
        ("effects", 4),
        ("result", 3),
        ("results", 3),
        ("leads", 3),
        ("lead", 3),
        ("consequence", 4),
        ("consequences", 4),
        ("mechanism", 4),
        ("mechanism", 4),
        ("explains", 3),
        ("explain", 2),
        ("due", 3),
    ],
    analogical: &[
        ("vs", 8),
        ("versus", 8),
        ("compare", 7),
        ("comparison", 7),
        ("difference", 6),
        ("differences", 6),
        ("similar", 5),
        ("similarity", 5),
        ("like", 4),
        ("unlike", 4),
        ("contrast", 5),
        ("better", 4),
        ("worse", 4),
        ("than", 3),
        ("alternative", 4),
    ],
    definitional: &[
        ("define", 7),
        ("definition", 7),
        ("what", 1),
        ("explain", 4),
        ("meaning", 6),
        ("means", 4),
        ("concept", 5),
        ("describe", 4),
        ("overview", 4),
        ("introduction", 3),
        ("about", 2),
    ],
    procedural: &[
        ("how", 5),
        ("steps", 7),
        ("step", 7),
        ("guide", 6),
        ("tutorial", 6),
        ("implement", 5),
        ("build", 4),
        ("create", 4),
        ("make", 3),
        ("install", 5),
        ("setup", 5),
        ("configure", 5),
        ("use", 3),
        ("usage", 4),
        ("do", 2),
    ],
    evaluative: &[
        ("good", 5),
        ("bad", 5),
        ("best", 6),
        ("worst", 6),
        ("better", 4),
        ("worth", 5),
        ("recommend", 6),
        ("should", 4),
        ("pros", 6),
        ("cons", 6),
        ("advantages", 5),
        ("disadvantages", 5),
        ("evaluate", 6),
        ("assessment", 5),
        ("opinion", 4),
        ("review", 4),
    ],
};

/// Stateless prompt classifier that maps natural-language input to a [`ReasoningMode`].
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::classifier::{InputClassifier, ReasoningMode};
///
/// assert_eq!(InputClassifier::classify("how to install Rust"), ReasoningMode::Procedural);
/// assert_eq!(InputClassifier::classify("rust vs go"), ReasoningMode::Analogical);
/// assert_eq!(InputClassifier::classify("photosynthesis"), ReasoningMode::Exploratory);
/// ```
pub struct InputClassifier;

impl InputClassifier {
    /// Classifies `prompt` into the most appropriate [`ReasoningMode`].
    ///
    /// Uses scored token matching: each word in the lowercased prompt is checked
    /// against signal-word tables for each mode; scores are summed and the
    /// highest-scoring mode is returned.  When all scores tie at zero (single
    /// word or unrecognised input) the result is [`ReasoningMode::Exploratory`].
    ///
    /// # Arguments
    ///
    /// * `prompt` - Raw user input string.
    ///
    /// # Time Complexity
    ///
    /// O(|prompt| × |signals|) = O(|prompt|) since signal tables are fixed-size.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::classifier::{InputClassifier, ReasoningMode};
    ///
    /// assert_eq!(InputClassifier::classify("what is the borrow checker?"), ReasoningMode::FactualLookup);
    /// assert_eq!(InputClassifier::classify("why is Rust fast?"), ReasoningMode::Causal);
    /// assert_eq!(InputClassifier::classify("is Rust worth learning?"), ReasoningMode::Evaluative);
    /// ```
    pub fn classify(prompt: &str) -> ReasoningMode {
        let words: Vec<String> = prompt
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_ascii_lowercase())
            .collect();

        if words.is_empty() {
            return ReasoningMode::Exploratory;
        }

        if words.len() == 1 {
            return ReasoningMode::Exploratory;
        }

        let score = |table: &[(&str, u32)]| -> u32 {
            words
                .iter()
                .map(|w| {
                    table
                        .iter()
                        .find(|(sig, _)| *sig == w.as_str())
                        .map(|(_, s)| *s)
                        .unwrap_or(0)
                })
                .sum()
        };

        let scores = [
            (ReasoningMode::Causal, score(SIGNALS.causal)),
            (ReasoningMode::Analogical, score(SIGNALS.analogical)),
            (ReasoningMode::Procedural, score(SIGNALS.procedural)),
            (ReasoningMode::Evaluative, score(SIGNALS.evaluative)),
            (ReasoningMode::Definitional, score(SIGNALS.definitional)),
            (ReasoningMode::FactualLookup, score(SIGNALS.factual)),
        ];

        let best = scores
            .iter()
            .max_by_key(|(_, s)| *s)
            .map(|(m, _)| *m)
            .unwrap_or(ReasoningMode::Exploratory);

        let best_score = scores.iter().map(|(_, s)| *s).max().unwrap_or(0);

        if best_score == 0 {
            ReasoningMode::Exploratory
        } else {
            best
        }
    }

    /// Returns `true` when the prompt is a bare noun or single-concept phrase
    /// with no question markers, indicating it should be treated as a
    /// [`ReasoningMode::Definitional`] or [`ReasoningMode::Exploratory`] query.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::classifier::InputClassifier;
    ///
    /// assert!(InputClassifier::is_bare_noun("photosynthesis"));
    /// assert!(InputClassifier::is_bare_noun("Rust programming language"));
    /// assert!(!InputClassifier::is_bare_noun("what is photosynthesis?"));
    /// ```
    pub fn is_bare_noun(prompt: &str) -> bool {
        static QUESTION_MARKERS: &[&str] = &[
            "what", "who", "when", "where", "why", "how", "which", "is", "are", "was", "were",
            "does", "do", "did", "can", "could", "would", "should", "will",
        ];
        let lower = prompt.to_ascii_lowercase();
        let first_word = lower
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_ascii_alphabetic());
        !QUESTION_MARKERS.contains(&first_word) && !prompt.contains('?')
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
