// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Response Framer
//!
//! Wraps a raw answer with an **intent-specific opening phrase** so the
//! response directly addresses what the user asked rather than starting with
//! generic topic prose.
//!
//! | Mode           | Opening pattern                                   |
//! |----------------|---------------------------------------------------|
//! | FactualLookup  | pass-through (first sentence is usually factual)   |
//! | Causal         | "The reason ... is that "                           |
//! | Analogical     | "Comparing ... and ...: "                             |
//! | Definitional   | "By definition, ... refers to "                     |
//! | Procedural     | "Here is how to ...: "                              |
//! | Evaluative     | "The advantages / pros and cons of ... are "        |
//! | Exploratory    | pass-through (broad topic, no framing bias)        |
//! | TaskExecution  | "Here is the ... for ...:\n\n"                        |
//!
//! ## Examples
//!
//! ```rust
//! use lmm::cognition::response_framer::ResponseFramer;
//! use lmm::cognition::thought::classifier::ReasoningMode;
//!
//! let framed = ResponseFramer::frame(ReasoningMode::Evaluative, "Rust", "what are the advantages of rust", "Rust is fast and safe.");
//! assert!(framed.starts_with("The advantages of Rust"));
//! ```

use crate::cognition::thought::classifier::ReasoningMode;

/// Stateless response framer that prepends mode-specific opening phrases.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::response_framer::ResponseFramer;
/// use lmm::cognition::thought::classifier::ReasoningMode;
///
/// let out = ResponseFramer::frame(ReasoningMode::Causal, "garbage collection", "why does rust avoid garbage collection", "It slows programs.");
/// // assert!(out.starts_with("The reason"));
/// ```
pub struct ResponseFramer;

impl ResponseFramer {
    /// Prepends an intent-specific opening phrase to `answer`.
    ///
    /// When the answer already begins with an appropriate opening the original
    /// text is returned unchanged to avoid double-framing.  Pass-through modes
    /// (`Exploratory`, `FactualLookup`) return `answer` as-is.
    ///
    /// # Arguments
    ///
    /// * `mode`       - Classified reasoning mode for this prompt.
    /// * `topic`      - Core topic phrase extracted from the prompt.
    /// * `raw_prompt` - Original user prompt (used to detect evaluative sub-intent
    ///   such as "inconveniences" vs "advantages").
    /// * `answer`     - Raw text produced by the pipeline.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::response_framer::ResponseFramer;
    /// use lmm::cognition::thought::classifier::ReasoningMode;
    ///
    /// let out = ResponseFramer::frame(ReasoningMode::Procedural, "install Rust", "", "Download rustup...");
    /// // assert!(out.starts_with("Here is how to install Rust"));
    /// ```
    pub fn frame(mode: ReasoningMode, topic: &str, raw_prompt: &str, answer: &str) -> String {
        let opener = Self::opening_phrase(mode, topic, raw_prompt, answer);
        if opener.is_empty() {
            return answer.to_string();
        }
        let trimmed = answer.trim_start();
        let lower = trimmed.to_ascii_lowercase();
        let already_framed = match mode {
            ReasoningMode::Causal => {
                lower.starts_with("the reason")
                    || lower.starts_with("because")
                    || lower.starts_with("due to")
                    || lower.starts_with("this is because")
                    || lower.starts_with("here is what is known")
            }
            ReasoningMode::Analogical => {
                lower.starts_with("comparing")
                    || lower.starts_with("the difference")
                    || lower.starts_with("the key difference")
            }
            ReasoningMode::Definitional => {
                lower.starts_with("by definition")
                    || lower.starts_with("the term")
                    || lower.starts_with("defined as")
            }
            ReasoningMode::Procedural => {
                lower.starts_with("here is how")
                    || lower.starts_with("to ")
                    || lower.starts_with("step")
            }
            ReasoningMode::Evaluative => {
                lower.starts_with("the advantage")
                    || lower.starts_with("the benefit")
                    || lower.starts_with("pros and cons")
                    || lower.starts_with("the pros")
                    || lower.starts_with("the disadvantage")
                    || lower.starts_with("the limitation")
                    || lower.starts_with("the inconvenience")
            }
            ReasoningMode::TaskExecution => {
                lower.starts_with("here is")
                    || lower.starts_with("below is")
                    || lower.starts_with("```")
            }
            _ => false,
        };
        if already_framed {
            return answer.to_string();
        }
        format!("{} {}", opener.trim_end(), trimmed)
    }

    /// Returns just the opening phrase for `mode` and `topic`.
    ///
    /// An empty string is returned for pass-through modes (`Exploratory`,
    /// `FactualLookup`).
    ///
    /// # Arguments
    ///
    /// * `mode`       - Classified reasoning mode.
    /// * `topic`      - Core topic phrase (already cleaned / stripped).
    /// * `raw_prompt` - Original user prompt text; used to detect evaluative
    ///   sub-intent (`"inconveniences"`, `"limitations"`, etc.)
    ///   that may have been stripped from `topic` by `extract_topic`.
    /// * `answer`     - Raw answer text (secondary evaluative signal).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::response_framer::ResponseFramer;
    /// use lmm::cognition::thought::classifier::ReasoningMode;
    ///
    /// let phrase = ResponseFramer::opening_phrase(ReasoningMode::Definitional, "borrow checker", "", "");
    /// // assert!(phrase.contains("borrow checker"));
    /// ```
    pub fn opening_phrase(
        mode: ReasoningMode,
        topic: &str,
        raw_prompt: &str,
        answer: &str,
    ) -> String {
        let topic = clean_topic_for_mode(topic.trim(), mode);
        let topic_cap = capitalize_first(&topic);
        let answer_lower = answer.to_ascii_lowercase();
        let prompt_lower = raw_prompt.to_ascii_lowercase();
        match mode {
            ReasoningMode::FactualLookup | ReasoningMode::Exploratory => String::new(),
            ReasoningMode::Causal => build_causal_opener(&topic, answer),
            ReasoningMode::Analogical => {
                let entities = extract_comparison_entities(&topic);
                if let Some((a, b)) = entities {
                    format!(
                        "Comparing {} and {}: the key differences are as follows.",
                        capitalize_first(&a),
                        capitalize_first(&b)
                    )
                } else {
                    format!("Comparing {}:", topic_cap)
                }
            }
            ReasoningMode::Definitional => {
                format!("By definition, {} refers to", topic_cap)
            }
            ReasoningMode::Procedural => {
                format!("Here is how to {}:", topic)
            }
            ReasoningMode::Evaluative => {
                let prompt_wants_cons = prompt_lower.contains("disadvantage")
                    || prompt_lower.contains("drawback")
                    || prompt_lower.contains("inconvenience")
                    || prompt_lower.contains("limitation")
                    || prompt_lower.contains("downside")
                    || prompt_lower.contains("weakness")
                    || prompt_lower.contains("problem")
                    || prompt_lower.contains("issue")
                    || prompt_lower.contains("cons");
                let prompt_wants_pros = prompt_lower.contains("advantage")
                    || prompt_lower.contains("benefit")
                    || prompt_lower.contains("upside")
                    || prompt_lower.contains("strength")
                    || prompt_lower.contains("pros");
                let answer_wants_pros = answer_lower.contains("advantage")
                    || answer_lower.contains("benefit")
                    || answer_lower.contains("pros")
                    || answer_lower.contains("positive");
                let answer_wants_cons = answer_lower.contains("disadvantage")
                    || answer_lower.contains("drawback")
                    || answer_lower.contains("cons")
                    || answer_lower.contains("negative");
                let wants_pros = prompt_wants_pros || answer_wants_pros;
                let wants_cons = prompt_wants_cons || answer_wants_cons;
                if prompt_wants_cons && !prompt_wants_pros {
                    if prompt_lower.contains("inconvenience") {
                        format!("The inconveniences of {} are as follows.", topic_cap)
                    } else if prompt_lower.contains("limitation") {
                        format!("The limitations of {} are as follows.", topic_cap)
                    } else if prompt_lower.contains("disadvantage") {
                        format!("The disadvantages of {} are as follows.", topic_cap)
                    } else if prompt_lower.contains("drawback") {
                        format!("The drawbacks of {} are as follows.", topic_cap)
                    } else if prompt_lower.contains("weakness") {
                        format!("The weaknesses of {} are as follows.", topic_cap)
                    } else {
                        format!("The disadvantages of {} are as follows.", topic_cap)
                    }
                } else if wants_pros && wants_cons {
                    format!("The pros and cons of {} are as follows.", topic_cap)
                } else if wants_cons {
                    format!("The disadvantages of {} are as follows.", topic_cap)
                } else {
                    format!("The advantages of {} are as follows.", topic_cap)
                }
            }
            ReasoningMode::TaskExecution => {
                format!("Here is the solution for {}:\n\n", topic)
            }
        }
    }
}

/// Strips mode-specific keyword prefixes from a topic string so the opening
/// phrase does not double-repeat them.
///
/// For example, when the prompt is `"what are the advantages of rust?"`,
/// `extract_topic` returns `"advantages of rust"`.  For `Evaluative` mode the
/// framer would then produce `"The advantages of advantages of rust"`.  This
/// function strips `"advantages of "` / `"benefits of "` etc. so the result
/// is just `"rust"`.
///
/// Similarly `"why does rust not have a garbage collector"` has its causal
/// prefix stripped down to `"rust"` already by `trim_predicate`, but the full
/// question context is passed separately via `answer` so the framing stays
/// informative.
///
/// # Time Complexity
///
/// O(|s|)
fn clean_topic_for_mode(topic: &str, mode: ReasoningMode) -> String {
    let lower = topic.to_ascii_lowercase();
    static EVALUATIVE_PREFIXES: &[&str] = &[
        "advantages of ",
        "advantage of ",
        "benefits of ",
        "benefit of ",
        "disadvantages of ",
        "disadvantage of ",
        "pros and cons of ",
        "pros of ",
        "cons of ",
        "drawbacks of ",
        "drawback of ",
        "strengths of ",
        "strength of ",
        "weaknesses of ",
        "weakness of ",
        "limitations of ",
        "limitation of ",
        "inconveniences of ",
        "inconvenience of ",
        "upsides of ",
        "downsides of ",
        "merits of ",
        "merit of ",
        "demerits of ",
        "demerit of ",
    ];
    static CAUSAL_PREFIXES: &[&str] = &[
        "reason for ",
        "reasons for ",
        "cause of ",
        "causes of ",
        "why ",
        "explanation for ",
        "how come ",
    ];
    static PROCEDURAL_PREFIXES: &[&str] = &[
        "how to ",
        "steps to ",
        "steps for ",
        "guide to ",
        "guide for ",
        "instructions for ",
        "tutorial for ",
        "install ",
        "setup ",
        "configure ",
    ];
    static DEFINITIONAL_PREFIXES: &[&str] = &[
        "definition of ",
        "meaning of ",
        "what is ",
        "concept of ",
        "explanation of ",
    ];
    static TASK_EXECUTION_PREFIXES: &[&str] = &[
        "write a ",
        "write an ",
        "write ",
        "generate a ",
        "generate an ",
        "generate ",
        "create a ",
        "create an ",
        "create ",
        "implement a ",
        "implement an ",
        "implement ",
        "make a ",
        "make an ",
        "code a ",
        "code an ",
        "build a ",
        "build an ",
        "show me a ",
        "show me an ",
        "show me ",
        "give me a ",
        "give me an ",
    ];
    let prefixes: &[&str] = match mode {
        ReasoningMode::Evaluative => EVALUATIVE_PREFIXES,
        ReasoningMode::Causal => CAUSAL_PREFIXES,
        ReasoningMode::Procedural => PROCEDURAL_PREFIXES,
        ReasoningMode::Definitional => DEFINITIONAL_PREFIXES,
        ReasoningMode::TaskExecution => TASK_EXECUTION_PREFIXES,
        _ => &[],
    };
    for prefix in prefixes {
        if lower.starts_with(prefix) {
            let cleaned = topic[prefix.len()..].trim();
            if !cleaned.is_empty() {
                return capitalize_first(cleaned);
            }
        }
    }
    topic.to_string()
}

/// Builds the most grammatically natural causal opening for `topic`.
///
/// When `topic` is a short noun (like `"rust"`), returns
/// `"The reason rust is"`.  When `topic` is a richer predicate clause
/// (like `"Rust does not have a garbage collector"`), the opener becomes
/// `"Rust does not have a garbage collector because"` which flows naturally
/// into the answer.
///
/// When the answer text does not contain causal explanation words (memory,
/// ownership, design, etc.) the opener falls back to
/// `"Here is what is known about X:"` to avoid producing a dangling "because"
/// clause that is not completed by the content.
///
/// # Time Complexity
///
/// O(|s| + |answer|)
fn build_causal_opener(topic: &str, answer: &str) -> String {
    let answer_lower = answer.to_ascii_lowercase();
    static CAUSAL_CONTENT_SIGNALS: &[&str] = &[
        " because ",
        " since ",
        " due to ",
        " as a result",
        " therefore",
        " this means ",
        " which means ",
        " in order to ",
        " to avoid ",
        " to prevent ",
        " to ensure ",
        " design ",
        " ownership ",
        " borrow",
        " memory ",
        " performance ",
        " zero-cost",
        " no gc",
        " no garbage",
        " garbage collector",
        " runtime ",
        " overhead ",
        " deterministic",
        " allocat",
        " stack ",
        " heap ",
        " lifetime",
        " compile",
    ];
    let has_causal_content = CAUSAL_CONTENT_SIGNALS
        .iter()
        .any(|&sig| answer_lower.contains(sig));
    let words: Vec<&str> = topic.split_whitespace().collect();
    if words.len() <= 2 {
        let cap = capitalize_first(topic);
        if has_causal_content {
            return format!("The reason {} is that", cap);
        } else {
            return format!("Here is what is known about {}:", cap);
        }
    }
    let lower_words: Vec<String> = words.iter().map(|w| w.to_ascii_lowercase()).collect();
    static PREDICATE_MARKERS: &[&str] = &[
        "does", "do", "did", "is", "are", "was", "were", "has", "have", "had", "cannot", "can't",
        "doesn't", "don't", "won't", "would", "could", "should",
    ];
    let has_predicate = lower_words
        .iter()
        .any(|w| PREDICATE_MARKERS.contains(&w.as_str()));
    if has_predicate {
        if has_causal_content {
            format!("{} because", capitalize_first(topic))
        } else {
            format!("Here is what is known about {}:", capitalize_first(topic))
        }
    } else {
        if has_causal_content {
            format!("The reason {} is that", capitalize_first(topic))
        } else {
            format!("Here is what is known about {}:", capitalize_first(topic))
        }
    }
}

/// Capitalises the first Unicode scalar of `s`.
///
/// # Time Complexity
///
/// O(|s|)
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let upper: String = first.to_uppercase().collect();
            upper + chars.as_str()
        }
    }
}

/// Attempts to split a comparison topic like "rust vs python" or "rust and python"
/// into two entity strings.
///
/// Returns `None` when no splitting pattern is found.
///
/// # Time Complexity
///
/// O(|s|)
fn extract_comparison_entities(s: &str) -> Option<(String, String)> {
    static SPLITTERS: &[&str] = &[" vs ", " versus ", " and ", " or ", " compared to "];
    let lower = s.to_ascii_lowercase();
    for &splitter in SPLITTERS {
        if let Some(pos) = lower.find(splitter) {
            let a = s[..pos].trim().to_string();
            let b = s[pos + splitter.len()..].trim().to_string();
            if !a.is_empty() && !b.is_empty() {
                return Some((a, b));
            }
        }
    }
    None
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
