// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # TRAM Web Fetch Constants

pub const MAX_LITE_RESULTS: usize = 10;
pub const MIN_PROVERB_WORD_COUNT: usize = 8;
pub const MAX_DUCKY_RESULTS: usize = 5;
pub const MIN_LITE_SNIPPET_WORD_COUNT: usize = 12;
pub const MIN_ENTITY_WORD_LENGTH: usize = 4;
pub const MAX_TOPIC_PAIR_WORDS: usize = 4;
pub const MIN_IA_WORD_COUNT: usize = 8;
pub const MAX_RELATED_TOPICS: usize = 5;
pub const MIN_IA_FALLBACK_WORD_COUNT: usize = 5;
pub const CAUSAL_OPENERS: &[&str] = &[
    "why does ",
    "why do ",
    "why is ",
    "why are ",
    "why was ",
    "why would ",
    "why should ",
    "why can't ",
    "why cannot ",
    "why doesn't ",
    "why don't ",
    "how come ",
];
pub const VS_SPLITTERS: &[&str] = &[" vs ", " versus "];

pub const EVALUATIVE_KEYWORDS: &[&str] = &[
    "advantage",
    "benefit",
    "pros",
    "disadvantage",
    "inconvenience",
    "limitation",
    "drawback",
    "weakness",
    "cons",
];

pub const CAUSAL_KEYWORDS: &[&str] = &[" cause", " reason"];

pub const PROGRAMMING_LANGUAGE_SUFFIX: &str = " programming language";

pub const ADVANTAGES_PREFIX: &str = "advantages of ";
pub const BENEFITS_PREFIX: &str = "benefits of ";
pub const DISADVANTAGES_PREFIX: &str = "disadvantages of ";
pub const PROS_CONS_PREFIX: &str = "pros and cons of ";
pub const WHAT_IS_PREFIX: &str = "what is ";

pub const MEMORY_MANAGEMENT_SUFFIX: &str = " memory management";
pub const OWNERSHIP_SUFFIX: &str = " ownership";
pub const DESIGN_DECISIONS_SUFFIX: &str = " design decisions";

pub const AMBIGUOUS_WORDS: &[&str] = &[
    "water", "ice", "fire", "air", "light", "sound", "time", "space",
];

pub const CHEMICAL_SUFFIX: &str = " chemical";
pub const COMPOUND_SUFFIX: &str = " compound";
pub const ELEMENT_SUFFIX: &str = " element";
pub const SCIENTIFIC_SUFFIX: &str = " scientific";

pub const DISADVANTAGE_KEYWORDS: &[&str] = &[
    "disadvantage",
    "inconvenience",
    "limitation",
    "drawback",
    "weakness",
    "cons",
];

pub const WEB_IA_SOURCE: &str = "web-ia";
pub const WEB_INTENT_SOURCE: &str = "web-intent";
pub const WEB_DUCKY_SOURCE: &str = "web-ducky";
pub const WEB_SCRAPE_SOURCE: &str = "web-scrape";
pub const WEB_IA_EVAL_SOURCE: &str = "web-ia-eval";
pub const WEB_IA_A_SOURCE: &str = "web-ia-a";
pub const WEB_IA_B_SOURCE: &str = "web-ia-b";

pub const AMBIGUOUS_LENGTH_THRESHOLD: usize = 5;
pub const MIN_WORD_COUNT_FOR_TEXT: usize = 8;

pub const DUCKY_PREFIX: &str = "!ducky+";

pub const WEB_LITE_SOURCE: &str = "web-lite";
pub const WEB_IA_FALLBACK_SOURCE: &str = "web-ia-fallback";

pub const COMPARE_PREFIX: &str = "compare ";
pub const AND_CONNECTOR: &str = " and ";
pub const DIFFERENCES_BETWEEN_PREFIX: &str = "differences between ";
pub const VS_CONNECTOR: &str = " vs ";

pub const DEFAULT_USER_AGENT: &str = "Mozilla/5.0";
pub const EMPTY_STRING: &str = "";
pub const FIREFOX_AGENT: &str = "firefox";

pub const OWNERSHIP_MODEL_SUFFIX: &str = " ownership model";
pub const ADVANTAGES_OF_PREFIX: &str = "advantages of ";
pub const PROS_CONS_SUFFIX: &str = " pros cons";
pub const BENEFITS_OF_PREFIX: &str = "benefits of ";

pub const DEEP_SCRAPE_SOURCE_LABEL: &str = "web-deep-scrape";
pub const MAX_DEEP_SCRAPE_ITERATIONS: usize = 5;
pub const MIN_DEEP_SCRAPE_RELEVANCE_SCORE: f64 = 0.3;
pub const MIN_DEEP_SCRAPE_CONTENT_LENGTH: usize = 100;

pub const DEEP_SCRAPE_KEYWORD_MIN_LENGTH: usize = 3;
pub const DEEP_SCRAPE_RELEVANCE_KEYWORD_MIN_LENGTH: usize = 4;
pub const DEEP_SCRAPE_MAX_RELATED_TOPICS: usize = 10;
pub const DEEP_SCRAPE_MAX_URLS_PER_ITERATION: usize = 5;
pub const DEEP_SCRAPE_KB_QUERY_LIMIT: usize = 3;
pub const DEEP_SCRAPE_IA_PARAM_SOFTWARE: &str = "software";
pub const DEEP_SCRAPE_IA_PARAM_WEB: &str = "web";
pub const DEEP_SCRAPE_DOMAIN_FILTER: &str = "duckduckgo.com";
pub const DEEP_SCRAPE_HTTP_PREFIX: &str = "http";

pub const HTML2TEXT_WIDTH: usize = 120;
pub const TEXT_FILTER_PATTERNS: &[&str] = &[
    "[[",
    "^{",
    "##",
    "[edit]",
    "[jump to",
    "download as pdf",
    "printable version",
    "in other projects",
    "from wikipedia",
    "look up",
    "log in",
    "move to sidebar",
    "toggle",
    "## contents",
    "## see also",
    "## references",
    "## external links",
    "## further reading",
    "## bibliography",
    "[citation needed]",
    "isbn",
    "* [",
    "==",
    "---",
];
pub const TEXT_FILTER_DOMAINS: &[&str] = &[
    "wikipedia",
    "wiktionary",
    "wikidata",
    "wikispecies",
    "wikivoyage",
];
pub const CITATION_REFERENCE_REGEX: &str = r"\^\{\[\[\d+\]\]\[\d+\]\}";
pub const CITATION_ARTIFACT_REGEX: &str = r"\[\[\d+\]\]\[\d+\]";
pub const STANDALONE_CITATION_REGEX: &str = r"\[\d+\]";
pub const CITATION_BRACKET_REGEX: &str = r"\[[^\]]+\]";
pub const MALFORMED_CITATION_REGEX: &str = r"\]\s*";

pub const URL_FILTER_DOMAINS: &[&str] = &[
    "duckduckgo.com",
    "wikimedia.org",
    "wikipedia.org/w/index.php",
];
pub const URL_FILTER_PATHS: &[&str] = &["/wiki/", "/donate", "/login", "/signup"];
pub const URL_FILTER_KEYWORDS: &[&str] = &["donate", "login", "signup"];
pub const URL_FILTER_SCHEMES: &[&str] = &["javascript:", "mailto:", "tel:", "ftp:"];

pub const SCRAPER_FILTER_DOMAINS: &[&str] = &["youtube.com", "youtu.be"];
pub const MAX_LINKED_URLS_TO_FOLLOW: usize = 4;

pub const DEEP_SCRAPE_CHUNK_SIZE: usize = 250;
pub const DEEP_SCRAPE_TOP_K_MULTIPLIER: usize = 3;
pub const DEEP_SCRAPE_MIN_RELEVANCE_THRESHOLD: f64 = 0.2;
