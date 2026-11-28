// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # TRAM Scraper Constants

pub const MIN_SENTENCE_LENGTH: usize = 8;
pub const MAX_PARAGRAPH_LENGTH: usize = 500;
pub const MIN_ENGLISH_RATIO: f64 = 0.7;

pub const MIN_DEEP_PARAGRAPH_WORDS: usize = 35;
pub const MIN_THINK_DEEP_PARAGRAPH_WORDS: usize = 25;
pub const MAX_DEEP_PARAGRAPHS_TO_SHOW: usize = 5;
pub const MAX_LINKED_URLS_TO_FOLLOW: usize = 5;

pub const EMPTY_GAP_PATTERN: &str = " , ";

/// Gap patterns that signal a Wikipedia template or citation was stripped,
/// leaving an orphaned article, preposition, or conjunction before punctuation.
pub const ARTICLE_GAP_PATTERNS: &[&str] = &[
    " the .",
    " a .",
    " an .",
    " the ,",
    " a ,",
    " an ,",
    " were ,",
    " was ,",
    " is ,",
    " are ,",
    " by ,",
    " in ,",
    " from ,",
    " or ,",
    " and ,",
    " to ,",
    " of ,",
    " by.",
    " in.",
    " from.",
    " or.",
    " to.",
    " of.",
    " by .",
    " in .",
    " from .",
    " or .",
    " to .",
    " of .",
    " a .",
    " an .",
    " the .",
    " and .",
    " with .",
    " at .",
    " on .",
    " using .",
    " either an",
    "either a ",
    " or a ,",
    " or an ,",
    " by as ",
    " by an ",
    " an and ",
    ", ,",
    "( ,",
    "( .",
];
pub const INCOMPLETE_PARENTHETICAL_PATTERNS: &[&str] = &[" (e.g.,", " (i.e.,"];
pub const INCOMPLETE_QUOTE_PATTERN: &str = " from \"";
pub const INCOMPLETE_QUOTE_EXCEPTION: &str = " from \"mostly";

pub const CITATION_SEMICOLON_GAP_PATTERN: &str = " or ;";
pub const CITATION_SEMICOLON_AND_GAP_PATTERN: &str = " and ;";
pub const CITATION_SEMICOLON_WAS_GAP_PATTERN: &str = " was ;";

pub const INCOMPLETE_THROUGH_PATTERN: &str = " designed to be highly through";
pub const INCOMPLETE_THROUGH_MODULES_PATTERN: &str = " highly through modules";
pub const INCOMPLETE_COMMON_PATTERN: &str = "a common in the";

/// CSS selector that matches HTML elements whose text content is treated as code.
///
/// Used by the scraper's code-block extraction pass before `html2text` strips tags.
pub const CODE_ELEMENT_SELECTOR: &str = "pre, code, textarea, .highlight, .codehilite, .code, .sourceCode, \
     .language-python, .language-javascript, .language-rust, .language-java, \
     .language-cpp, .language-c, .language-bash, .language-shell";

/// CSS selector that matches a CodeMirror editor container element.
///
/// Sites like pythontutorial.net, programiz.com, and tutorialspoint.com render code
/// inside CodeMirror widgets.  The editor stores each line in a separate
/// `<pre class="CodeMirror-line">` element inside a `<div class="CodeMirror-code">`.
/// This selector is used to locate the container so all per-line `<pre>` children
/// can be joined into a single multi-line code block before ingestion.
pub const CODEMIRROR_CODE_SELECTOR: &str = ".CodeMirror-code";

/// Fallback CSS selector when [`CODE_ELEMENT_SELECTOR`] parsing fails.
pub const FALLBACK_CODE_SELECTOR: &str = "pre";

/// CSS selector for per-line pre elements inside CodeMirror widgets.
pub const CODEMIRROR_LINE_SELECTOR: &str = "pre";

/// CSS selector matching main article and prose paragraphs in parsed HTML.
pub const HTML_PROSE_SELECTOR: &str = "p, article, main, .content, .post-content, .entry-content";

/// Fallback CSS selector matching the entire document body.
pub const HTML_BODY_SELECTOR: &str = "body";

/// Domains to skip when scraping web pages.
pub const YOUTUBE_FILTER_DOMAINS: &[&str] = &["youtube.com", "youtu.be"];

/// Multiplier for ascii code punctuation count when checking for mathematical prose in HTML.
pub const HTML_MATH_PROSE_MULTIPLIER: usize = 2;

/// Opening delimiter for GitHub-Flavored Markdown fenced code blocks.
pub const FENCE_OPEN: &str = "```";

/// Maximum lines kept from a single fenced code block found in plain text.
pub const MAX_FENCE_LINES: usize = 60;

pub const CODE_KEYWORDS: &[&str] = &[
    "block expression",
    "semicolon",
    "fn main()",
    "println!",
    "let mut",
];
pub const CODE_INDICATORS: &[&str] = &["`", "pure }", "typestates"];
pub const TABLE_INDICATORS: &[&str] = &["│", "|"];

pub const SENTENCE_END_INCOMPLETE: &[&str] = &[" the ", " of ", " for "];
pub const STANDALONE_PUNCTUATION: &[&str] = &["}", ")"];

/// Markers that identify Wikipedia-style disambiguation sentences.
///
/// Any sentence from the deep-scraped corpus that contains one of these
/// substrings (case-insensitive) is stripped before the response is assembled,
/// preventing topic-disambiguation noise from leaking into the answer.
pub const DISAMBIGUATION_MARKERS: &[&str] = &[
    " a song by ",
    " a film by ",
    " an album by ",
    " a novel by ",
    " an ep by ",
    " a television ",
    " a tv series",
    " a single by ",
    " a video game ",
    " may refer to",
    " can refer to",
    "see also:",
    "(disambiguation)",
    " redirect here",
];

/// Substrings indicating a bot-detection or access-denied page.
///
/// Paragraphs containing any of these strings (case-insensitive) are rejected
/// by the deep-paragraph filter to prevent bot-block pages from leaking into
/// the final response.
pub const BOT_DETECTION_PATTERNS: &[&str] = &[
    "malicious software",
    "automated requests",
    "browser plug-in",
    "ip address may be",
    "verify you are human",
    "access denied",
    "rate limit exceeded",
    "unusual traffic",
    "captcha",
    "our systems have detected",
    "this request was blocked",
    "too many requests",
];

/// URL path suffixes that identify synthetically-generated Wikipedia pages
/// that almost certainly do not exist (e.g. `Water_algorithm`,
/// `Water_program`).  URLs containing these patterns are excluded from the
/// deep-search URL list and the response URL footer.
pub const URL_SYNTHETIC_WIKI_SUFFIXES: &[&str] = &[
    "_program",
    "_algorithm",
    "_method",
    "_function",
    "_computation",
    "_example",
    "_implementation",
];

/// Prefix pattern for synthetically-generated "Numerical_X" Wikipedia URLs.
///
/// Must be all-lowercase because it is compared against `url.to_ascii_lowercase()`.
pub const URL_SYNTHETIC_NUMERICAL_PREFIX: &str = "/wiki/numerical_";

pub const MONTH_PATTERNS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
