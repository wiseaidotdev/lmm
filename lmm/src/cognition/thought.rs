// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Adaptive Thought Engine (ATE)
//!
//! This module implements a **novel, CPU-only, training-free reasoning system**
//! that generates a unique internal dialogue for every user request before the
//! existing [`crate::cognition::r#loop::ThinkLoop`] runs.
//!
//! ## Architecture
//!
//! ```text
//! User Prompt
//!   │
//!   ├─► InputClassifier          → ReasoningMode
//!   │
//!   ├─► GoalDecomposer           → ordered sub-goals
//!   │
//!   ├─► ThoughtChain (loop):
//!   │     ├─ formulate premise from prior step
//!   │     ├─ oracle fetch → evidence
//!   │     ├─ choose StepMode (Deductive / Abductive / Analogical / Corrective)
//!   │     ├─ build ThoughtStep with BeliefInterval
//!   │     ├─ update DS-combined belief
//!   │     └─ InternalCritic → if failure detected, insert Corrective step
//!   │
//!   └─► ThoughtChainResult
//!         ├─ insight_text          (English prose distilled from the chain)
//!         ├─ prioritized_queries   (refined queries for ThinkLoop)
//!         └─ confidence            (DS-combined final belief midpoint)
//! ```
//!
//! ## Sub-modules
//!
//! | Module | Key Type | Role |
//! |---|---|---|
//! | `belief` | [`belief::BeliefInterval`] | Dempster-Shafer uncertainty intervals |
//! | `classifier` | [`classifier::InputClassifier`] | Scores prompt → ReasoningMode |
//! | `critic` | [`critic::InternalCritic`] | Detects contradictions / stagnation |
//! | `decompose` | [`decompose::GoalDecomposer`] | Expands goal into sub-goals |
//!
//! ## Reasoning Modes
//!
//! The engine adapts its internal monologue to seven modes:
//! `FactualLookup`, `Causal`, `Analogical`, `Definitional`,
//! `Procedural`, `Exploratory`, `Evaluative`.
//!
//! ## Examples
//!
//! ```rust
//! use lmm::cognition::thought::AdaptiveThoughtEngine;
//! use lmm::cognition::search::SearchOracle;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut oracle = SearchOracle::new(5);
//!     let engine = AdaptiveThoughtEngine::new(6, 0.55, 2);
//!     let result = engine.run("what is Rust?", &mut oracle).await;
//!     assert!(!result.insight_text.is_empty());
//!     assert!(!result.prioritized_queries.is_empty());
//! }
//! ```
//!
//! ## See Also
//!
//! * [Dempster-Shafer theory](https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory)
//! * [Abductive reasoning](https://en.wikipedia.org/wiki/Abductive_reasoning)

pub mod belief;
pub mod classifier;
pub mod critic;
pub mod decompose;

use crate::cognition::search::SearchOracle;
use crate::cognition::signal::error_from_texts;
use crate::cognition::thought::{
    belief::BeliefInterval,
    classifier::{InputClassifier, ReasoningMode},
    critic::{CritiqueResult, InternalCritic, StepView},
    decompose::GoalDecomposer,
};

/// The reasoning mode applied to derive a single [`ThoughtStep`].
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::StepMode;
///
/// assert_eq!(StepMode::Deductive.label(), "deductive");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMode {
    /// Evidence strongly and directly supports the conclusion.
    Deductive,
    /// Evidence partially matches; the best explanation is inferred.
    Abductive,
    /// Conclusion is drawn by analogy from a related domain.
    Analogical,
    /// Corrects or revises a prior step detected as flawed by the critic.
    Corrective,
}

impl StepMode {
    /// Returns a short lowercase label for this mode.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Deductive => "deductive",
            Self::Abductive => "abductive",
            Self::Analogical => "analogical",
            Self::Corrective => "corrective",
        }
    }
}

/// A single step in the internal thought chain.
///
/// Each step records the **premise** (what we know), the **transition reasoning**
/// (the logical bridge), the **conclusion** (what we derived), the supporting
/// [`BeliefInterval`], and the raw **evidence text** extracted from the oracle.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::{ThoughtStep, StepMode};
/// use lmm::cognition::thought::belief::BeliefInterval;
///
/// let step = ThoughtStep {
///     index: 0,
///     premise: "The user asked about Rust.".to_string(),
///     reasoning: "Rust is a known systems programming language.".to_string(),
///     conclusion: "I should start by defining what Rust is.".to_string(),
///     belief: BeliefInterval::new(0.7, 0.9),
///     mode: StepMode::Deductive,
///     evidence_text: "Rust is a systems programming language focused on safety.".to_string(),
/// };
/// assert_eq!(step.mode, StepMode::Deductive);
/// ```
#[derive(Debug, Clone)]
pub struct ThoughtStep {
    /// Zero-based step index in the chain.
    pub index: usize,
    /// What we know or observed that motivates this step.
    pub premise: String,
    /// The logical or inferential bridge from premise to conclusion (for display/logging only).
    pub reasoning: String,
    /// What we concluded from applying the reasoning to the premise.
    pub conclusion: String,
    /// Dempster-Shafer belief interval supporting this conclusion.
    pub belief: BeliefInterval,
    /// The reasoning mode applied in this step.
    pub mode: StepMode,
    /// Raw factual text extracted from the oracle (suitable for KB ingestion).
    pub evidence_text: String,
}

/// The complete internal reasoning chain produced by [`AdaptiveThoughtEngine`].
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::{ThoughtChain};
/// use lmm::cognition::thought::classifier::ReasoningMode;
///
/// let chain = ThoughtChain::new("what is rust?", ReasoningMode::FactualLookup);
/// assert!(chain.steps.is_empty());
/// assert_eq!(chain.mode, ReasoningMode::FactualLookup);
/// ```
#[derive(Debug, Clone)]
pub struct ThoughtChain {
    /// The original user prompt this chain is reasoning about.
    pub goal: String,
    /// The detected reasoning mode governing this chain.
    pub mode: ReasoningMode,
    /// Ordered sequence of reasoning steps.
    pub steps: Vec<ThoughtStep>,
    /// The ordered sub-goals derived from the goal decomposition.
    pub sub_goals: Vec<String>,
    /// The running DS-combined belief across all steps.
    pub final_belief: BeliefInterval,
}

impl ThoughtChain {
    /// Constructs a new empty chain for `goal` under `mode`.
    pub fn new(goal: impl Into<String>, mode: ReasoningMode) -> Self {
        Self {
            goal: goal.into(),
            mode,
            steps: Vec::new(),
            sub_goals: Vec::new(),
            final_belief: BeliefInterval::ignorant(),
        }
    }

    /// Appends a step and updates the running DS-combined belief.
    pub fn push_step(&mut self, step: ThoughtStep) {
        self.final_belief = self.final_belief.combine(&step.belief);
        self.steps.push(step);
    }

    /// Returns the last step's conclusion, or the goal when the chain is empty.
    pub fn last_conclusion(&self) -> &str {
        self.steps
            .last()
            .map(|s| s.conclusion.as_str())
            .unwrap_or(&self.goal)
    }

    /// Distils the chain into a coherent English prose summary.
    ///
    /// Concatenates the reasoning field from each step, separated by spaces, to
    /// form a natural flowing thought sequence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::{ThoughtChain, ThoughtStep, StepMode};
    /// use lmm::cognition::thought::belief::BeliefInterval;
    /// use lmm::cognition::thought::classifier::ReasoningMode;
    ///
    /// let mut chain = ThoughtChain::new("test", ReasoningMode::Exploratory);
    /// chain.push_step(ThoughtStep {
    ///     index: 0,
    ///     premise: "p".to_string(),
    ///     reasoning: "First observation leads to an insight.".to_string(),
    ///     conclusion: "c".to_string(),
    ///     belief: BeliefInterval::new(0.6, 0.8),
    ///     mode: StepMode::Deductive,
    ///     evidence_text: "First observation leads to an insight.".to_string(),
    /// });
    /// let text = chain.distil();
    /// assert!(text.contains("First observation"));
    /// ```
    pub fn distil(&self) -> String {
        self.steps
            .iter()
            .map(|s| s.reasoning.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extracts only the factual evidence content from the chain for KB ingestion.
    ///
    /// Unlike `distil()`, this method collects the raw `evidence_text` field from
    /// each step, the actual words retrieved from the oracle, without any
    /// meta-reasoning commentary.  This prevents thought templates
    /// (e.g. "Following from what I already know...") from leaking into the KB
    /// and surfacing verbatim in final responses.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::{ThoughtChain, ThoughtStep, StepMode};
    /// use lmm::cognition::thought::belief::BeliefInterval;
    /// use lmm::cognition::thought::classifier::ReasoningMode;
    ///
    /// let mut chain = ThoughtChain::new("test", ReasoningMode::Exploratory);
    /// chain.push_step(ThoughtStep {
    ///     index: 0,
    ///     premise: "p".to_string(),
    ///     reasoning: "Following from what I know, the evidence confirms that X is true.".to_string(),
    ///     conclusion: "c".to_string(),
    ///     belief: BeliefInterval::new(0.6, 0.8),
    ///     mode: StepMode::Deductive,
    ///     evidence_text: "X is a well-known property of Y systems.".to_string(),
    /// });
    /// let insight = chain.factual_insight();
    /// assert!(insight.contains("X is a well-known"));
    /// assert!(!insight.contains("Following from"));
    /// ```
    pub fn factual_insight(&self) -> String {
        self.steps
            .iter()
            .filter_map(|s| {
                let t = s.evidence_text.trim();
                let wc = t.split_whitespace().count();
                if (6..=60).contains(&wc) && !is_link_title(t) {
                    Some(t)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// The complete output of a single [`AdaptiveThoughtEngine::run`] invocation.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::ThoughtChainResult;
/// use lmm::cognition::thought::classifier::ReasoningMode;
///
/// let result = ThoughtChainResult {
///     chain: lmm::cognition::thought::ThoughtChain::new("test", ReasoningMode::Exploratory),
///     prioritized_queries: vec!["test overview".to_string()],
///     insight_text: "Test is an important concept.".to_string(),
///     factual_insight: "Test is an important concept.".to_string(),
///     confidence: 0.75,
/// };
/// assert_eq!(result.confidence, 0.75);
/// ```
#[derive(Debug, Clone)]
pub struct ThoughtChainResult {
    /// The full reasoning chain.
    pub chain: ThoughtChain,
    /// Sub-goals re-ordered by uncertainty, ready for `ThinkLoop` queries.
    pub prioritized_queries: Vec<String>,
    /// English prose distilled from all step reasoning fields (for logging/verbose display only).
    pub insight_text: String,
    /// Pure factual evidence text extracted from oracle responses (safe for KB ingestion).
    pub factual_insight: String,
    /// DS-combined belief midpoint across all steps ∈ [0, 1].
    pub confidence: f64,
}

/// CPU-only adaptive reasoning engine that generates a unique thought chain per prompt.
///
/// Build via [`AdaptiveThoughtEngine::new`] or [`AdaptiveThoughtEngine::default`],
/// then call [`AdaptiveThoughtEngine::run`] to obtain a [`ThoughtChainResult`].
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::thought::AdaptiveThoughtEngine;
/// use lmm::cognition::search::SearchOracle;
///
/// #[tokio::main]
/// async fn main() {
///     let engine = AdaptiveThoughtEngine::default();
///     let mut oracle = SearchOracle::new(3);
///     let result = engine.run("photosynthesis", &mut oracle).await;
///     assert!(!result.prioritized_queries.is_empty());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveThoughtEngine {
    /// Maximum thought steps to generate (iteration cap).
    pub max_thought_steps: usize,
    /// DS-interval midpoint threshold above which the chain stops early.
    pub min_belief_threshold: f64,
    /// Stagnation patience passed to [`InternalCritic`].
    pub critic_patience: usize,
}

impl AdaptiveThoughtEngine {
    /// Constructs a new engine with explicit parameters.
    ///
    /// # Arguments
    ///
    /// * `max_thought_steps`    - Maximum steps to generate (≥ 1).
    /// * `min_belief_threshold` - Early-stop threshold ∈ [0, 1].
    /// * `critic_patience`      - Steps without improvement before stagnation fires.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::thought::AdaptiveThoughtEngine;
    ///
    /// let engine = AdaptiveThoughtEngine::new(8, 0.55, 2);
    /// assert_eq!(engine.max_thought_steps, 8);
    /// ```
    pub fn new(
        max_thought_steps: usize,
        min_belief_threshold: f64,
        critic_patience: usize,
    ) -> Self {
        Self {
            max_thought_steps: max_thought_steps.max(1),
            min_belief_threshold: min_belief_threshold.clamp(0.0, 1.0),
            critic_patience: critic_patience.max(1),
        }
    }

    /// Runs the adaptive thought chain for `prompt` and returns a [`ThoughtChainResult`].
    ///
    /// ## Algorithm
    ///
    /// 1. Classify the prompt → [`ReasoningMode`].
    /// 2. Decompose the goal into sub-goals via [`GoalDecomposer`].
    /// 3. For each sub-goal (up to `max_thought_steps`):
    ///    a. Formulate a premise from the prior step's conclusion.
    ///    b. Fetch evidence from `oracle`.
    ///    c. Derive a [`BeliefInterval`] from token-recall overlap.
    ///    d. Choose a [`StepMode`]: Deductive if recall > 0.6, Abductive if 0.3-0.6,
    ///    Analogical if mode is `Analogical`, Corrective if critic fired.
    ///    e. Build a [`ThoughtStep`] and push it into the chain.
    ///    f. Run [`InternalCritic`], insert a corrective step if needed.
    ///    g. Stop early if belief is actionable.
    /// 4. Prioritise sub-goals by uncertainty.
    /// 5. Distil the chain into `insight_text`.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Raw user input.
    /// * `oracle` - `SearchOracle` to fetch evidence from (uses cache; no duplicate requests).
    ///
    /// # Time Complexity
    ///
    /// O(`max_thought_steps` × |oracle_response|)
    pub async fn run(&self, prompt: &str, oracle: &mut SearchOracle) -> ThoughtChainResult {
        let mode = InputClassifier::classify(prompt);
        let sub_goals = GoalDecomposer::decompose(prompt, mode, &[]);
        let critic = InternalCritic::new(0.02, self.critic_patience);

        let mut chain = ThoughtChain::new(prompt, mode);
        chain.sub_goals = sub_goals.clone();

        let mut evidenced_beliefs: Vec<BeliefInterval> = Vec::new();
        let mut corrective_pending: Option<CritiqueResult> = None;

        for (step_idx, sub_goal) in sub_goals.iter().enumerate() {
            if step_idx >= self.max_thought_steps {
                break;
            }

            let premise = self.build_premise(&chain, sub_goal, mode, step_idx);
            let query = self.build_query(&chain, sub_goal);
            let evidence = oracle.fetch(&query).await;

            let goal_tokens = count_tokens(prompt);
            let evidence_tokens = count_overlap(prompt, &evidence);
            let raw_belief = BeliefInterval::from_evidence(evidence_tokens, goal_tokens.max(1));

            let step_mode = if corrective_pending.is_some() {
                corrective_pending = None;
                StepMode::Corrective
            } else {
                self.choose_step_mode(mode, raw_belief.mid(), step_idx)
            };

            let evidence_text =
                extract_key_phrase(&evidence, &GoalDecomposer::extract_topic(prompt, &[]))
                    .unwrap_or_default();

            let reasoning = self.build_reasoning(
                &chain,
                sub_goal,
                &evidence_text,
                step_mode,
                raw_belief.mid(),
                step_idx,
            );
            let conclusion =
                self.build_conclusion(sub_goal, &evidence_text, step_mode, raw_belief.mid());

            let step = ThoughtStep {
                index: step_idx,
                premise: premise.clone(),
                reasoning,
                conclusion,
                belief: raw_belief.clone(),
                mode: step_mode,
                evidence_text,
            };

            chain.push_step(step);
            evidenced_beliefs.push(raw_belief);

            let views: Vec<StepView> = chain
                .steps
                .iter()
                .map(|s| StepView {
                    index: s.index,
                    premise: &s.premise,
                    conclusion: &s.conclusion,
                    belief: s.belief.clone(),
                })
                .collect();

            if let Some(critique) = critic.evaluate(&views) {
                corrective_pending = Some(critique);
            }

            if chain.final_belief.is_actionable(self.min_belief_threshold) {
                break;
            }
        }

        let prioritized_queries = GoalDecomposer::prioritize(sub_goals, &evidenced_beliefs);

        let insight_text = chain.distil();
        let factual_insight = chain.factual_insight();
        let confidence = chain.final_belief.mid();

        ThoughtChainResult {
            chain,
            prioritized_queries,
            insight_text,
            factual_insight,
            confidence,
        }
    }

    /// Formulates the premise for step `idx` based on the prior step's conclusion.
    fn build_premise(
        &self,
        chain: &ThoughtChain,
        sub_goal: &str,
        mode: ReasoningMode,
        idx: usize,
    ) -> String {
        if idx == 0 {
            format!(
                "The user input '{}' is classified as a {} query. I should explore: {}.",
                chain.goal,
                mode.label(),
                sub_goal
            )
        } else {
            let prior = chain.last_conclusion();
            format!("{} Building on this, I now examine: {}.", prior, sub_goal)
        }
    }

    /// Builds the oracle query for a given sub-goal and chain context.
    fn build_query(&self, chain: &ThoughtChain, sub_goal: &str) -> String {
        let topic = GoalDecomposer::extract_topic(&chain.goal, &[]);
        if sub_goal.contains(topic.as_str()) {
            sub_goal.to_string()
        } else {
            format!("{} {}", topic, sub_goal)
        }
    }

    /// Chooses the most appropriate [`StepMode`] for the current step.
    fn choose_step_mode(&self, mode: ReasoningMode, recall: f64, idx: usize) -> StepMode {
        if mode == ReasoningMode::Analogical && idx > 0 {
            return StepMode::Analogical;
        }
        if recall >= 0.55 {
            StepMode::Deductive
        } else {
            StepMode::Abductive
        }
    }

    /// Builds an English-prose reasoning sentence for a step.
    fn build_reasoning(
        &self,
        chain: &ThoughtChain,
        sub_goal: &str,
        evidence_text: &str,
        step_mode: StepMode,
        recall: f64,
        idx: usize,
    ) -> String {
        let topic = GoalDecomposer::extract_topic(&chain.goal, &[]);
        let has_evidence = !evidence_text.trim().is_empty();

        match step_mode {
            StepMode::Deductive => {
                if has_evidence {
                    let snippet: String = evidence_text
                        .split_whitespace()
                        .take(20)
                        .collect::<Vec<_>>()
                        .join(" ");
                    if idx == 0 {
                        format!(
                            "The evidence establishes that {}. This gives me a foundation to explore {}.",
                            snippet, sub_goal
                        )
                    } else {
                        format!(
                            "Building on prior context, the evidence confirms: {}.",
                            snippet
                        )
                    }
                } else {
                    format!(
                        "Reasoning from first principles about '{}': this is a core aspect of {}.",
                        sub_goal, topic
                    )
                }
            }
            StepMode::Abductive => {
                if has_evidence {
                    format!(
                        "The evidence partially supports this aspect: {}.",
                        evidence_text
                    )
                } else {
                    format!(
                        "Evidence on '{}' is limited (recall {:.0}%). This aspect requires further investigation.",
                        sub_goal,
                        recall * 100.0
                    )
                }
            }
            StepMode::Analogical => {
                if has_evidence {
                    format!(
                        "Drawing a comparison: {}. This illuminates key patterns in '{}'.",
                        evidence_text, sub_goal
                    )
                } else {
                    format!(
                        "By analogy, '{}' shares structural properties with known patterns in related domains.",
                        sub_goal
                    )
                }
            }
            StepMode::Corrective => {
                if has_evidence {
                    format!("Reconsidering '{}': {}.", topic, evidence_text)
                } else {
                    format!(
                        "Re-examining '{}' from a different angle reveals this aspect needs broader context.",
                        topic
                    )
                }
            }
        }
    }

    /// Builds a conclusion sentence for the current step.
    fn build_conclusion(
        &self,
        sub_goal: &str,
        evidence_text: &str,
        step_mode: StepMode,
        recall: f64,
    ) -> String {
        let has_evidence = !evidence_text.trim().is_empty();
        match step_mode {
            StepMode::Deductive => {
                if has_evidence {
                    format!("Established: {}.", evidence_text)
                } else {
                    format!("Established the importance of exploring: {}.", sub_goal)
                }
            }
            StepMode::Abductive => {
                if recall > 0.3 && has_evidence {
                    format!("Best explanation: {}.", evidence_text)
                } else {
                    format!(
                        "Insufficient evidence for '{}', further exploration needed.",
                        sub_goal
                    )
                }
            }
            StepMode::Analogical => format!(
                "By analogy, '{}' shares structural properties with known patterns.",
                sub_goal
            ),
            StepMode::Corrective => {
                if has_evidence {
                    format!("Revised understanding: {}.", evidence_text)
                } else {
                    format!("Revised: '{}' requires a broader perspective.", sub_goal)
                }
            }
        }
    }
}

impl Default for AdaptiveThoughtEngine {
    fn default() -> Self {
        Self::new(6, 0.55, 2)
    }
}

/// Counts the total number of meaningful tokens (> 2 chars) in `text`.
fn count_tokens(text: &str) -> usize {
    text.split_whitespace()
        .filter(|w| w.len() > 2)
        .count()
        .max(1)
}

/// Counts how many tokens from `goal` appear in `observation` (case-insensitive).
fn count_overlap(goal: &str, observation: &str) -> usize {
    if observation.is_empty() {
        return 0;
    }
    let recall = error_from_texts(goal, observation);
    let goal_tokens = count_tokens(goal);
    ((1.0 - recall) * goal_tokens as f64).round() as usize
}

/// Returns `true` when `s` looks like a web-page title list rather than natural-language prose.
///
/// Heuristics applied (any match returns `true`):
/// - Contains separator patterns that indicate navigation strings: `" - "`, `" | "`, `" \u00b7 "`.
/// - Very high ratio of title-case words (> 75%) among four or more words.
/// - Presence of a year token (2020-2030) together with high title-case ratio (> 50%).
fn is_link_title(s: &str) -> bool {
    if s.contains(" - ") || s.contains(" | ") || s.contains(" \u{00b7} ") {
        return true;
    }
    let trimmed = s.trim();
    if trimmed.starts_with('-') || trimmed.starts_with('•') || trimmed.starts_with('*') {
        return true;
    }
    if s.contains('[') && s.contains(']') {
        return true;
    }
    let colon_count = s.chars().filter(|&c| c == ':').count();
    if colon_count >= 2 {
        return true;
    }
    static SITE_NAMES: &[&str] = &[
        "geeksforgeeks",
        "stackoverflow",
        "reddit",
        "medium.com",
        "github",
        "quora",
        "dev.to",
        "freecodecamp",
        "tutorialspoint",
        "programiz",
        "towardsdatascience",
        "hackernews",
    ];
    let lower = s.to_ascii_lowercase();
    for site in SITE_NAMES {
        if lower.contains(site) {
            return true;
        }
    }
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }
    let title_case_count = words
        .iter()
        .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        .count();
    let ratio = title_case_count as f64 / words.len() as f64;
    if ratio > 0.75 && words.len() >= 4 {
        return true;
    }
    let has_year = words.iter().any(|w| {
        if w.len() != 4 || !w.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        let n: u32 = w.parse().unwrap_or(0);
        (2020..=2030).contains(&n)
    });
    has_year && ratio > 0.5
}

/// Extracts a representative factual key phrase from `evidence` by selecting the
/// best natural-language prose sentence mentioning `topic` (case-insensitive).
///
/// # Prioritisation Order
///
/// 1. Sentences containing `topic` that pass the [`is_link_title`] filter.
/// 2. Any prose sentence (not a link title) with >= 5 words.
/// 3. `None` when the evidence is empty or entirely link-title junk.
///
/// # Arguments
///
/// * `evidence` - Raw text fetched from the oracle.
/// * `topic`    - Core topic word(s) to preference-match against.
fn extract_key_phrase(evidence: &str, topic: &str) -> Option<String> {
    if evidence.trim().is_empty() {
        return None;
    }
    let lower_topic = topic.to_ascii_lowercase();

    let sentences: Vec<&str> = evidence
        .split(['.', '!', '?', '\n'])
        .map(|s| s.trim())
        .filter(|s| {
            let wc = s.split_whitespace().count();
            (5..=50).contains(&wc)
        })
        .collect();

    let prose: Vec<&&str> = sentences.iter().filter(|s| !is_link_title(s)).collect();

    let topic_hit = prose
        .iter()
        .filter(|s| s.to_ascii_lowercase().contains(&lower_topic))
        .max_by_key(|s| s.split_whitespace().count())
        .copied();

    if let Some(s) = topic_hit {
        let words: Vec<&str> = s.split_whitespace().take(25).collect();
        return Some(words.join(" "));
    }

    let any_hit = prose
        .iter()
        .max_by_key(|s| s.split_whitespace().count())
        .copied();

    if let Some(s) = any_hit {
        let words: Vec<&str> = s.split_whitespace().take(25).collect();
        return Some(words.join(" "));
    }

    None
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
