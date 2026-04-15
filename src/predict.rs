use crate::discovery::SymbolicRegression;
use crate::equation::Expression;
use crate::error::{LmmError, Result};
use crate::lexicon::{Lexicon, word_tone};
use std::collections::HashMap;

static ARTICLES: &[&str] = &["the", "a", "an"];
static PREPOSITIONS: &[&str] = &[
    "in", "of", "through", "beyond", "within", "across", "beneath", "among", "into", "from",
    "toward", "over", "between", "after", "before", "along", "behind", "around", "upon",
];
static CONJUNCTIONS: &[&str] = &[
    "and", "yet", "or", "while", "because", "where", "that", "which", "though", "whereas", "since",
    "as",
];
static COPULAS: &[&str] = &[
    "is", "was", "are", "were", "have", "had", "would", "could", "may", "must", "shall", "should",
    "can",
];

static COMMON_NOUNS: &[&str] = &[
    "world",
    "life",
    "time",
    "mind",
    "heart",
    "soul",
    "truth",
    "light",
    "force",
    "power",
    "knowledge",
    "form",
    "space",
    "nature",
    "order",
    "energy",
    "field",
    "depth",
    "path",
    "wave",
    "core",
    "edge",
    "flow",
    "source",
    "point",
    "thought",
    "reason",
    "vision",
    "sense",
    "voice",
    "code",
    "pattern",
    "motion",
    "change",
    "state",
    "law",
    "ground",
    "center",
    "realm",
    "layer",
    "base",
    "frame",
    "structure",
    "system",
    "process",
    "signal",
    "curve",
    "node",
    "origin",
    "principle",
    "concept",
    "theory",
    "equation",
    "formula",
    "rhythm",
    "harmony",
    "logic",
    "relation",
    "measure",
    "value",
    "symbol",
    "meaning",
    "wisdom",
    "insight",
    "clarity",
    "beauty",
    "shape",
    "dimension",
    "symmetry",
    "balance",
    "proportion",
    "ratio",
    "scale",
    "scope",
    "proof",
    "creation",
    "design",
    "purpose",
    "legacy",
    "cosmos",
    "matter",
    "element",
    "charge",
    "pulse",
    "cycle",
    "basis",
    "language",
    "message",
    "craft",
    "substance",
    "presence",
    "silence",
    "movement",
    "boundary",
    "horizon",
    "current",
    "gravity",
    "tension",
    "density",
    "volume",
    "surface",
    "fabric",
    "memory",
    "future",
    "history",
    "moment",
    "age",
    "epoch",
    "dawn",
];

static COMMON_ADJECTIVES: &[&str] = &[
    "ancient", "deep", "vast", "pure", "clear", "bright", "true", "real", "great", "strong",
    "long", "wide", "high", "free", "open", "dark", "light", "still", "calm", "bold", "new", "old",
    "known", "hidden", "sacred", "cosmic", "eternal", "infinite", "divine", "natural", "logical",
    "formal", "primal", "human", "living", "moving", "rising", "central", "vital", "basic",
    "complex", "simple", "ordered", "precise", "exact", "linear", "dynamic", "static", "global",
    "total", "subtle", "dense", "outer", "inner", "silent", "primal", "finite", "higher", "noble",
    "fluid", "solid", "curved", "broken", "woven", "resonant", "latent",
];

static COMMON_VERBS: &[&str] = &[
    "reveal",
    "encode",
    "create",
    "form",
    "hold",
    "reach",
    "grow",
    "rise",
    "flows",
    "moves",
    "builds",
    "shapes",
    "contains",
    "reflects",
    "extends",
    "emerges",
    "unfolds",
    "expands",
    "compresses",
    "expresses",
    "represents",
    "models",
    "captures",
    "defines",
    "describes",
    "governs",
    "enables",
    "transforms",
    "generates",
    "solves",
    "proves",
    "measures",
    "encodes",
    "connects",
    "binds",
    "weaves",
    "follows",
    "guides",
    "aligns",
];

static COMMON_ADVERBS: &[&str] = &[
    "deeply",
    "clearly",
    "truly",
    "freely",
    "greatly",
    "widely",
    "highly",
    "fully",
    "still",
    "always",
    "often",
    "never",
    "only",
    "merely",
    "precisely",
    "exactly",
    "naturally",
    "silently",
    "eternally",
];

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CoarsePos {
    Article,
    Preposition,
    Conjunction,
    Verb,
    Noun,
    Adjective,
    Adverb,
    Unknown,
}

fn detect_pos(word: &str) -> CoarsePos {
    let w = word.to_ascii_lowercase();
    match w.as_str() {
        "the" | "a" | "an" => CoarsePos::Article,
        "in" | "of" | "to" | "for" | "with" | "at" | "by" | "from" | "into" | "on" | "upon"
        | "through" | "over" | "under" | "between" | "among" | "during" | "before" | "after"
        | "above" | "below" | "beside" | "beyond" | "within" | "across" | "toward" | "along"
        | "behind" | "around" => CoarsePos::Preposition,
        "and" | "but" | "or" | "nor" | "so" | "yet" | "although" | "because" | "if" | "since"
        | "though" | "unless" | "until" | "when" | "where" | "while" | "that" | "which" | "who"
        | "whereas" | "as" => CoarsePos::Conjunction,
        "is" | "are" | "was" | "were" | "be" | "been" | "have" | "has" | "had" | "do" | "does"
        | "did" | "will" | "would" | "shall" | "should" | "may" | "might" | "must" | "can"
        | "could" => CoarsePos::Verb,
        _ => {
            if COMMON_NOUNS.contains(&w.as_str()) {
                CoarsePos::Noun
            } else if COMMON_ADJECTIVES.contains(&w.as_str()) {
                CoarsePos::Adjective
            } else if COMMON_VERBS.contains(&w.as_str()) {
                CoarsePos::Verb
            } else if COMMON_ADVERBS.contains(&w.as_str()) || w.ends_with("ly") {
                CoarsePos::Adverb
            } else if w.ends_with("tion")
                || w.ends_with("ness")
                || w.ends_with("ity")
                || w.ends_with("ment")
                || w.ends_with("ics")
                || w.ends_with("ism")
                || w.ends_with("ence")
                || w.ends_with("ance")
                || w.ends_with("dom")
                || w.ends_with("hood")
            {
                CoarsePos::Noun
            } else if w.ends_with("ful")
                || w.ends_with("less")
                || w.ends_with("ical")
                || w.ends_with("ous")
                || w.ends_with("ive")
                || w.ends_with("ible")
                || w.ends_with("able")
                || w.ends_with("ic")
                || w.ends_with("al")
            {
                CoarsePos::Adjective
            } else if w.ends_with("ing")
                || w.ends_with("ize")
                || w.ends_with("ise")
                || w.ends_with("es")
                || w.ends_with("ed")
            {
                CoarsePos::Verb
            } else {
                CoarsePos::Unknown
            }
        }
    }
}

fn expected_next_pos(history: &[CoarsePos], step: usize) -> CoarsePos {
    match history.last().copied() {
        Some(CoarsePos::Article) => CoarsePos::Adjective,
        Some(CoarsePos::Adjective) => CoarsePos::Noun,
        Some(CoarsePos::Noun) => match step % 5 {
            0 => CoarsePos::Verb,
            1 => CoarsePos::Preposition,
            2 => CoarsePos::Conjunction,
            3 => CoarsePos::Verb,
            _ => CoarsePos::Adverb,
        },
        Some(CoarsePos::Verb) => match step % 2 {
            0 => CoarsePos::Article,
            _ => CoarsePos::Preposition,
        },
        Some(CoarsePos::Preposition) => CoarsePos::Article,
        Some(CoarsePos::Conjunction) => CoarsePos::Article,
        Some(CoarsePos::Adverb) => CoarsePos::Adjective,
        Some(CoarsePos::Unknown) => CoarsePos::Preposition,
        None => CoarsePos::Article,
    }
}

fn is_function_pos(pos: CoarsePos) -> bool {
    matches!(
        pos,
        CoarsePos::Article | CoarsePos::Preposition | CoarsePos::Conjunction | CoarsePos::Verb
    )
}

fn pick_function_word(pos: CoarsePos, recency: &HashMap<String, usize>) -> &'static str {
    let candidates: &[&str] = match pos {
        CoarsePos::Article => ARTICLES,
        CoarsePos::Preposition => PREPOSITIONS,
        CoarsePos::Conjunction => CONJUNCTIONS,
        CoarsePos::Verb => COPULAS,
        _ => ARTICLES,
    };
    candidates
        .iter()
        .min_by_key(|&&w| recency.get(w).copied().unwrap_or(0))
        .copied()
        .unwrap_or("the")
}

fn suffix_match(window: &[String], recent: &[String], max_suffix: usize) -> Option<String> {
    for len in (2..=max_suffix.min(recent.len())).rev() {
        let suffix = &recent[recent.len() - len..];
        for pos in 0..window.len().saturating_sub(len) {
            if window[pos..pos + len] == *suffix && pos + len < window.len() {
                return Some(window[pos + len].clone());
            }
        }
    }
    None
}

fn evaluate_at(eq: &Expression, pos: f64) -> f64 {
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), pos);
    eq.evaluate(&vars).unwrap_or(107.0)
}

fn stable_target_tone(eq: &Expression, pos: f64, context_tones: &[f64]) -> f64 {
    let mean = context_tones.iter().sum::<f64>() / context_tones.len().max(1) as f64;
    let gp = evaluate_at(eq, pos);

    (if !gp.is_finite() || (gp - mean).abs() > 6.0 {
        mean
    } else {
        gp
    })
    .clamp(97.0, 122.0)
}

fn stable_target_length(eq: &Expression, pos: f64) -> usize {
    let raw = evaluate_at(eq, pos).round();
    if raw.is_finite() && (3.0..=10.0).contains(&raw) {
        raw as usize
    } else {
        5
    }
}

fn score_word(
    word: &str,
    target_tone: f64,
    target_length: usize,
    recency: &HashMap<String, usize>,
) -> f64 {
    let r = recency.get(word).copied().unwrap_or(0) as f64 * 6.0;
    let tone_diff = (word_tone(word) - target_tone).abs();
    let len_diff = (word.len() as f64 - target_length as f64).abs() * 0.8;
    r + tone_diff + len_diff
}

fn best_from_pool<'a>(
    pool: &[&'a str],
    target_tone: f64,
    target_length: usize,
    recency: &HashMap<String, usize>,
) -> Option<&'a str> {
    pool.iter().copied().min_by(|&a, &b| {
        let sa = score_word(a, target_tone, target_length, recency);
        let sb = score_word(b, target_tone, target_length, recency);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    })
}

pub struct TextPredictor {
    pub window_size: usize,
    pub iterations: usize,
    pub depth: usize,
    pub lexicon: Option<Lexicon>,
}

impl TextPredictor {
    pub fn new(window_size: usize, iterations: usize, depth: usize) -> Self {
        Self {
            window_size,
            iterations,
            depth,
            lexicon: None,
        }
    }

    pub fn with_lexicon(mut self, lexicon: Lexicon) -> Self {
        self.lexicon = Some(lexicon);
        self
    }

    fn fit_trajectory(&self, positions: &[f64], tones: &[f64]) -> Result<Expression> {
        if positions.len() < 2 {
            return Err(LmmError::Discovery("Need at least 2 tokens".into()));
        }
        let inputs: Vec<Vec<f64>> = positions.iter().map(|&p| vec![p]).collect();
        SymbolicRegression::new(self.depth, self.iterations)
            .with_variables(vec!["x".into()])
            .with_population(60)
            .fit(&inputs, tones)
    }

    fn fit_rhythm(&self, positions: &[f64], lengths: &[f64]) -> Result<Expression> {
        if positions.len() < 2 {
            return Err(LmmError::Discovery("Need at least 2 tokens".into()));
        }
        let inputs: Vec<Vec<f64>> = positions.iter().map(|&p| vec![p]).collect();
        SymbolicRegression::new(self.depth.min(3), self.iterations / 2)
            .with_variables(vec!["x".into()])
            .with_population(40)
            .fit(&inputs, lengths)
    }

    fn curated_pool_for_pos(next_pos: CoarsePos) -> &'static [&'static str] {
        match next_pos {
            CoarsePos::Noun => COMMON_NOUNS,
            CoarsePos::Adjective => COMMON_ADJECTIVES,
            CoarsePos::Verb => COMMON_VERBS,
            CoarsePos::Adverb => COMMON_ADVERBS,
            _ => COMMON_NOUNS,
        }
    }

    fn select_content_word(
        &self,
        next_pos: CoarsePos,
        trajectory_eq: &Expression,
        rhythm_eq: &Expression,
        pos: f64,
        recency: &HashMap<String, usize>,
        context_tones: &[f64],
    ) -> String {
        let target_tone = stable_target_tone(trajectory_eq, pos, context_tones);
        let target_length = stable_target_length(rhythm_eq, pos);

        let curated = Self::curated_pool_for_pos(next_pos);
        if let Some(w) = best_from_pool(curated, target_tone, target_length, recency) {
            return w.to_string();
        }

        if let Some(lexicon) = &self.lexicon {
            let candidates = lexicon.candidates(target_length, target_tone, 2, 8.0, 30);
            if let Some(w) = best_from_pool(&candidates, target_tone, target_length, recency) {
                return w.to_string();
            }
        }

        "world".to_string()
    }

    fn select_from_context(
        &self,
        window_tokens: &[String],
        trajectory_eq: &Expression,
        rhythm_eq: &Expression,
        pos: f64,
        recency: &HashMap<String, usize>,
        context_tones: &[f64],
    ) -> String {
        let target_tone = stable_target_tone(trajectory_eq, pos, context_tones);
        let target_len = stable_target_length(rhythm_eq, pos);
        let refs: Vec<&str> = window_tokens.iter().map(String::as_str).collect();
        best_from_pool(&refs, target_tone, target_len, recency)
            .unwrap_or("world")
            .to_string()
    }

    pub fn predict_continuation(
        &self,
        text: &str,
        predict_length: usize,
    ) -> Result<PredictedContinuation> {
        if text.is_empty() {
            return Err(LmmError::Perception("Input text is empty".into()));
        }
        let all_tokens: Vec<String> = text.split_whitespace().map(String::from).collect();
        if all_tokens.len() < 2 {
            return Err(LmmError::Perception("Need at least 2 words".into()));
        }
        let window_start = all_tokens.len().saturating_sub(self.window_size);
        let window_tokens = &all_tokens[window_start..];

        let positions: Vec<f64> = (0..window_tokens.len()).map(|i| i as f64).collect();
        let tones: Vec<f64> = window_tokens
            .iter()
            .map(|t| word_tone(&t.to_ascii_lowercase()))
            .collect();
        let lengths: Vec<f64> = window_tokens.iter().map(|t| t.len() as f64).collect();

        let trajectory_eq = self.fit_trajectory(&positions, &tones)?;
        let rhythm_eq = self.fit_rhythm(&positions, &lengths)?;

        let mut pos_history: Vec<CoarsePos> = window_tokens.iter().map(|t| detect_pos(t)).collect();
        let mut suffix_context: Vec<String> =
            window_tokens[window_tokens.len().saturating_sub(3)..].to_vec();
        let mut continuation = String::new();
        let mut pos = window_tokens.len() as f64;
        let mut recency: HashMap<String, usize> = HashMap::new();
        let mut step: usize = 0;

        while continuation.len() < predict_length {
            let suffix_result =
                suffix_match(window_tokens, &suffix_context, 2.min(suffix_context.len()));

            let chosen = match suffix_result {
                Some(w) => w,
                None => {
                    let next_pos = expected_next_pos(&pos_history, step);
                    if is_function_pos(next_pos) {
                        pick_function_word(next_pos, &recency).to_string()
                    } else if self.lexicon.is_some() {
                        self.select_content_word(
                            next_pos,
                            &trajectory_eq,
                            &rhythm_eq,
                            pos,
                            &recency,
                            &tones,
                        )
                    } else {
                        self.select_from_context(
                            window_tokens,
                            &trajectory_eq,
                            &rhythm_eq,
                            pos,
                            &recency,
                            &tones,
                        )
                    }
                }
            };

            continuation.push(' ');
            continuation.push_str(&chosen);
            *recency.entry(chosen.clone()).or_insert(0) += 1;
            pos_history.push(detect_pos(&chosen));
            if pos_history.len() > 8 {
                pos_history.remove(0);
            }
            suffix_context.push(chosen);
            if suffix_context.len() > 3 {
                suffix_context.remove(0);
            }
            pos += 1.0;
            step += 1;
        }

        Ok(PredictedContinuation {
            trajectory_equation: trajectory_eq,
            rhythm_equation: rhythm_eq,
            window_used: window_tokens.len(),
            continuation,
        })
    }
}

pub struct PredictedContinuation {
    pub trajectory_equation: Expression,
    pub rhythm_equation: Expression,
    pub window_used: usize,
    pub continuation: String,
}
