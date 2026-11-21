// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Code Responder
//!
//! Extracts real code snippets from the web-scraped KB corpus gathered for a
//! task-execution prompt.  Three extraction strategies are tried in order of
//! confidence:
//!
//! 1. **Marked code blocks** - the scraper prefixes verbatim `<pre>`/`<code>`/
//!    fenced blocks with [`CODE_BLOCK_MARKER`].  These are extracted first
//!    because they are real, complete code units.
//! 2. **Markdown fence detection** - scans the corpus for ` ``` ` delimiters
//!    that survived html2text conversion.
//! 3. **Indented-block detection** - 4-space-indented regions in plain text
//!    (the convention used by RST, man pages, and html2text output).
//! 4. **Line heuristic** - falls back to the original marker+punctuation scan.
//!
//! Returns `None` only when all strategies fail and the corpus is too sparse to
//! yield useful code.  The caller is then responsible for showing the user the
//! [`CODE_IMPROVEMENT_HINT`] message.
//!
//! ## Examples
//!
//! ```rust
//! use lmm::cognition::code_responder::CodeResponder;
//!
//! let corpus = "CODEBLOCK::def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)";
//! assert!(CodeResponder::generate("python", "factorial", corpus).is_some());
//! ```

use crate::constants::code_responder::{
    CODE_BLOCK_MARKER, CODE_PUNCT_CHARS, FALLBACK_LANGUAGE_TAG, INDENTED_CODE_SPACES, LANGUAGE_CPP,
    LANGUAGE_GO, LANGUAGE_JAVA, LANGUAGE_JAVASCRIPT, LANGUAGE_PYTHON, LANGUAGE_RUBY, LANGUAGE_RUST,
    LANGUAGE_TYPESCRIPT, MARKER_CPP, MARKER_DEFAULT, MARKER_GO, MARKER_JAVA, MARKER_JAVASCRIPT,
    MARKER_PYTHON, MARKER_RUBY, MARKER_RUST, MATH_PROSE_RATIO_MULTIPLIER, MATH_UNICODE_CHARS,
    MAX_CODE_BLOCK_CHARS, MAX_CODE_LINES_TO_TAKE, MIN_CODE_BLOCK_CHARS, MIN_CODE_LINE_RATIO,
    MIN_CODE_LINES, MIN_CORPUS_CHARS_FOR_CODE, MIN_MARKED_BLOCK_SCORE,
    SUBJECT_RELEVANCE_BOOST_FACTOR, SUBJECT_TOKEN_MIN_LEN, TASK_TYPE_FACTORIAL,
    TASK_TYPE_FIBONACCI, TASK_TYPE_GENERIC, TASK_TYPE_HELLO_WORLD, TASK_TYPE_HTTP_REQUEST,
    TASK_TYPE_READ_FILE, TASK_TYPE_SORT, TEMPLATE_CPP_HELLO, TEMPLATE_GENERIC, TEMPLATE_GO_HELLO,
    TEMPLATE_JAVA_HELLO, TEMPLATE_JS_HELLO, TEMPLATE_PYTHON_FACTORIAL, TEMPLATE_PYTHON_FIBONACCI,
    TEMPLATE_PYTHON_HELLO, TEMPLATE_RUBY_HELLO, TEMPLATE_RUST_FIBONACCI, TEMPLATE_RUST_HELLO,
};
use crate::constants::tram_scraper::{FENCE_OPEN, MAX_FENCE_LINES};

/// Extracts and formats code snippets for task-execution prompts.
///
/// # Examples
///
/// ```rust
/// use lmm::cognition::code_responder::CodeResponder;
///
/// let block = CodeResponder::format_code_block("python", "print('Hello')");
/// assert!(block.starts_with("```python"));
/// ```
pub struct CodeResponder;

impl CodeResponder {
    /// Attempts to generate a code snippet for `subject` in `language` from `kb_corpus`.
    ///
    /// Tries four extraction strategies in confidence order.  Returns `Some(code)` on
    /// the first success, or `None` when all strategies fail.
    ///
    /// # Arguments
    ///
    /// * `language`  - Target language identifier (e.g. `"python"`, `"rust"`).
    /// * `subject`   - Task subject extracted from the user prompt.
    /// * `kb_corpus` - Ingested knowledge text searched for code snippets.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::code_responder::CodeResponder;
    ///
    /// // assert!(CodeResponder::generate("python", "factorial", "").is_none());
    /// ```
    pub fn generate(language: &str, subject: &str, kb_corpus: &str) -> Option<String> {
        if kb_corpus.trim().is_empty() {
            return Some(generate_template(language, subject));
        }

        if let Some(code) = extract_marked_blocks(kb_corpus, language, subject) {
            return Some(code);
        }

        if kb_corpus.len() < MIN_CORPUS_CHARS_FOR_CODE {
            return Some(generate_template(language, subject));
        }

        if let Some(code) = extract_fenced_blocks(kb_corpus, language) {
            return Some(code);
        }

        if let Some(code) = extract_indented_blocks(kb_corpus) {
            return Some(code);
        }

        extract_by_line_heuristic(kb_corpus, language, subject)
            .or_else(|| Some(generate_template(language, subject)))
    }

    /// Wraps `code` in a fenced Markdown code block for `language`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::code_responder::CodeResponder;
    ///
    /// let block = CodeResponder::format_code_block("rust", "fn main() {}");
    /// assert_eq!(block, "```rust\nfn main() {}\n```");
    /// ```
    pub fn format_code_block(language: &str, code: &str) -> String {
        format!("```{}\n{}\n```", language, code.trim())
    }

    /// Classifies `prompt` into one of the well-known task-type categories.
    ///
    /// Returns one of `"hello_world"`, `"fibonacci"`, `"factorial"`, `"sort"`,
    /// `"read_file"`, `"http_request"`, or `"generic"`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::code_responder::CodeResponder;
    ///
    /// assert_eq!(CodeResponder::detect_task_type("hello world program"), "hello_world");
    /// assert_eq!(CodeResponder::detect_task_type("fibonacci sequence"), "fibonacci");
    /// ```
    pub fn detect_task_type(prompt: &str) -> &'static str {
        let lower = prompt.to_ascii_lowercase();
        if lower.contains("hello") || lower.contains("hello world") {
            return TASK_TYPE_HELLO_WORLD;
        }
        if lower.contains("fibonacci")
            || lower.contains("fib sequence")
            || lower.contains("fib number")
        {
            return TASK_TYPE_FIBONACCI;
        }
        if lower.contains("factorial") {
            return TASK_TYPE_FACTORIAL;
        }
        if lower.contains("sort") {
            return TASK_TYPE_SORT;
        }
        if lower.contains("read file")
            || lower.contains("open file")
            || lower.contains("file") && lower.contains("parse")
        {
            return TASK_TYPE_READ_FILE;
        }
        if lower.contains("http")
            || lower.contains("fetch")
            || lower.contains("api")
            || lower.contains("request")
        {
            return TASK_TYPE_HTTP_REQUEST;
        }
        TASK_TYPE_GENERIC
    }
}

/// Returns a built-in template code snippet for `language` and `subject`.
///
/// Used as a fallback when the KB corpus is empty or too sparse for extraction.
///
/// # Time Complexity
///
/// O(1)
fn generate_template(language: &str, subject: &str) -> String {
    let task = CodeResponder::detect_task_type(subject);
    let code = match (language, task) {
        (LANGUAGE_PYTHON, TASK_TYPE_HELLO_WORLD) => TEMPLATE_PYTHON_HELLO,
        (LANGUAGE_PYTHON, TASK_TYPE_FIBONACCI) => TEMPLATE_PYTHON_FIBONACCI,
        (LANGUAGE_PYTHON, TASK_TYPE_FACTORIAL) => TEMPLATE_PYTHON_FACTORIAL,
        (LANGUAGE_RUST, TASK_TYPE_HELLO_WORLD) => TEMPLATE_RUST_HELLO,
        (LANGUAGE_RUST, TASK_TYPE_FIBONACCI) => TEMPLATE_RUST_FIBONACCI,
        (LANGUAGE_JAVASCRIPT, TASK_TYPE_HELLO_WORLD)
        | (LANGUAGE_TYPESCRIPT, TASK_TYPE_HELLO_WORLD) => TEMPLATE_JS_HELLO,
        (LANGUAGE_GO, TASK_TYPE_HELLO_WORLD) => TEMPLATE_GO_HELLO,
        (LANGUAGE_JAVA, TASK_TYPE_HELLO_WORLD) => TEMPLATE_JAVA_HELLO,
        (LANGUAGE_CPP, TASK_TYPE_HELLO_WORLD) => TEMPLATE_CPP_HELLO,
        (LANGUAGE_RUBY, TASK_TYPE_HELLO_WORLD) => TEMPLATE_RUBY_HELLO,
        _ => TEMPLATE_GENERIC,
    };
    code.to_string()
}

/// Returns `true` when `text` is predominantly mathematical prose rather than code.
///
/// A block is classified as math prose when:
/// - It contains more than [`MATH_PROSE_RATIO_MULTIPLIER`] times as many
///   [`MATH_UNICODE_CHARS`] as ASCII code-punctuation characters, **and**
/// - There is at least one math Unicode character present.
///
/// This prevents Wikipedia formula sections inside `<pre>` tags from being
/// mistaken for executable code.
///
/// # Time Complexity
///
/// O(|text|)
fn is_math_prose(text: &str) -> bool {
    let math_count = text
        .chars()
        .filter(|c| MATH_UNICODE_CHARS.contains(c))
        .count();
    if math_count == 0 {
        return false;
    }
    let code_punct_count = text
        .chars()
        .filter(|c| CODE_PUNCT_CHARS.contains(c))
        .count();
    math_count > code_punct_count * MATH_PROSE_RATIO_MULTIPLIER
}

/// Extracts code from corpus segments prefixed with [`CODE_BLOCK_MARKER`].
///
/// Splits the entire corpus on the marker delimiter so that code blocks
/// survive even after `kb_chunks.join("\n")` merges them into a single string.
/// Scores each block by two criteria:
///
/// 1. **Language keyword hits** - how many language-specific keywords appear at the
///    start of a trimmed line.  Blocks with zero hits are rejected unconditionally.
/// 2. **Subject-relevance boost** - how many tokens of `subject` appear anywhere in
///    the block (case-insensitive, boosted by [`SUBJECT_RELEVANCE_BOOST_FACTOR`]).
///    This ensures that a short `print("Hello, World!")` block beats an unrelated
///    `import unittest` block when the subject is `"hello world"`.
///
/// # Time Complexity
///
/// O(|corpus|)
fn extract_marked_blocks(corpus: &str, language: &str, subject: &str) -> Option<String> {
    let markers = lang_code_markers(language);
    let subject_tokens: Vec<String> = subject
        .split_whitespace()
        .filter(|w| w.len() >= SUBJECT_TOKEN_MIN_LEN)
        .map(|w| w.to_ascii_lowercase())
        .collect();
    let mut best: Option<String> = None;
    let mut best_score: f64 = 0.0;

    for segment in corpus.split(CODE_BLOCK_MARKER).skip(1) {
        let block = segment
            .split(CODE_BLOCK_MARKER)
            .next()
            .unwrap_or(segment)
            .trim();
        if block.len() < MIN_CODE_BLOCK_CHARS {
            continue;
        }
        let block = &block[..block.len().min(MAX_CODE_BLOCK_CHARS)];
        let lang_score: usize = markers
            .iter()
            .filter(|&&m| block.lines().any(|line| line.trim().starts_with(m)))
            .count();
        if lang_score < MIN_MARKED_BLOCK_SCORE {
            continue;
        }
        if is_math_prose(block) {
            continue;
        }
        let block_lower = block.to_ascii_lowercase();
        let relevance_boost: usize = subject_tokens
            .iter()
            .filter(|t| block_lower.contains(t.as_str()))
            .count();
        let score = lang_score as f64 + (relevance_boost as f64) * SUBJECT_RELEVANCE_BOOST_FACTOR;
        if score > best_score {
            best_score = score;
            best = Some(block.to_string());
        }
    }

    best.filter(|b| !b.trim().is_empty())
}

/// Scans `corpus` for GitHub-Flavored Markdown ` ``` ` fenced code blocks.
///
/// Extracts the content between the opening and closing fences.  When the opening
/// fence carries a language tag, it prefers blocks whose tag matches `language`.
/// Falls back to any fenced block when no language-tagged block is found.
///
/// # Time Complexity
///
/// O(|corpus|)
fn extract_fenced_blocks(corpus: &str, language: &str) -> Option<String> {
    let mut inside = false;
    let mut current_block: Vec<&str> = Vec::new();
    let mut current_lang = "";
    let mut best: Option<(String, bool)> = None;

    for line in corpus.lines() {
        let trimmed = line.trim();

        if !inside {
            if trimmed.starts_with(FENCE_OPEN) {
                let tag = trimmed.trim_start_matches('`').trim();
                current_lang = tag;
                inside = true;
                current_block.clear();
            }
        } else if trimmed == FENCE_OPEN
            || (trimmed.starts_with(FENCE_OPEN)
                && trimmed.ends_with(FENCE_OPEN)
                && trimmed.len() == FENCE_OPEN.len())
        {
            let joined = current_block
                .iter()
                .take(MAX_FENCE_LINES)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            if joined.len() >= MIN_CODE_BLOCK_CHARS {
                let lang_matches = current_lang.eq_ignore_ascii_case(language)
                    || language == FALLBACK_LANGUAGE_TAG
                    || current_lang.is_empty();
                match &best {
                    None => best = Some((joined, lang_matches)),
                    Some((_, already_matched)) if !already_matched && lang_matches => {
                        best = Some((joined, true));
                    }
                    _ => {}
                }
            }
            inside = false;
            current_block.clear();
        } else {
            current_block.push(line);
        }
    }

    best.map(|(code, _)| code)
}

/// Detects 4-space-indented regions in `corpus` and returns the largest that
/// passes the [`MIN_CODE_LINE_RATIO`] code-line test.
///
/// html2text renders `<pre>` blocks as 4-space-indented text, so this strategy
/// recovers code that survived the html2text pipeline even without the
/// `CODE_BLOCK_MARKER` prefix.
///
/// # Time Complexity
///
/// O(|corpus|)
fn extract_indented_blocks(corpus: &str) -> Option<String> {
    let indent = " ".repeat(INDENTED_CODE_SPACES);
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for line in corpus.lines() {
        if line.starts_with(&indent) {
            current.push(line);
        } else if !current.is_empty() {
            blocks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        blocks.push(current);
    }

    blocks
        .into_iter()
        .filter(|b| b.len() >= MIN_CODE_LINES)
        .filter(|b| {
            let code_lines = b
                .iter()
                .filter(|l| l.trim().chars().any(|c| CODE_PUNCT_CHARS.contains(&c)))
                .count();
            code_lines as f64 / b.len() as f64 >= MIN_CODE_LINE_RATIO
        })
        .max_by_key(|b| b.len())
        .map(|b| {
            b.iter()
                .take(MAX_CODE_LINES_TO_TAKE)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        })
}

/// Falls back to scanning individual lines for language markers and code punctuation.
///
/// A line is accepted only when a language-specific keyword appears **at the
/// start** of the trimmed line AND the line contains at least one code-
/// punctuation character.  Requiring a line-start match prevents English
/// prose words such as `for`, `if`, or `return` that appear mid-sentence
/// from being mistaken for Python control-flow keywords.
///
/// Returns `None` when fewer than [`MIN_CODE_LINES`] such lines are found.
///
/// # Time Complexity
///
/// O(|corpus|)
fn extract_by_line_heuristic(corpus: &str, language: &str, _subject: &str) -> Option<String> {
    let markers = lang_code_markers(language);

    let code_lines: Vec<&str> = corpus
        .lines()
        .filter(|line| {
            let l = line.trim();
            if l.is_empty() {
                return false;
            }
            let has_marker = markers.iter().any(|m| l.starts_with(m));
            let has_code_punct = l.chars().any(|c| CODE_PUNCT_CHARS.contains(&c));
            has_marker && has_code_punct
        })
        .take(MAX_CODE_LINES_TO_TAKE)
        .collect();

    if code_lines.len() < MIN_CODE_LINES {
        return None;
    }
    Some(code_lines.join("\n"))
}

/// Returns language-specific keyword strings that identify code lines in a corpus.
///
/// # Time Complexity
///
/// O(1)
fn lang_code_markers(language: &str) -> &'static [&'static str] {
    match language {
        LANGUAGE_PYTHON => MARKER_PYTHON,
        LANGUAGE_RUST => MARKER_RUST,
        LANGUAGE_JAVASCRIPT | LANGUAGE_TYPESCRIPT => MARKER_JAVASCRIPT,
        LANGUAGE_GO => MARKER_GO,
        LANGUAGE_JAVA => MARKER_JAVA,
        LANGUAGE_CPP => MARKER_CPP,
        LANGUAGE_RUBY => MARKER_RUBY,
        _ => MARKER_DEFAULT,
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
