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

pub const EMPTY_STRING: &str = "";

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
pub const DEEP_SCRAPE_DDG_REDIRECT_PREFIX: &str = "/l/?uddg=";
pub const DEEP_SCRAPE_DDG_REDIRECT_PARAM: &str = "uddg=";

/// Log message emitted when DDG Lite returns an empty result list for an
/// iteration.  When this appears in `--verbose` output the deep-search
/// fallback relies solely on the IA JSON API (`abstract_url`, `definition_url`,
/// `related_topics`) to discover URLs.
pub const DEEP_SCRAPE_NO_RESULTS_MSG: &str =
    "lite search returned no results, relying on IA API URLs";

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
    "... show more",
    "… show more",
    "…show more",
    "show more.",
    "show less",
    "read more",
    "see more",
    "load more",
    "click to expand",
    "expand all",
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

pub const SCRAPER_FILTER_DOMAINS: &[&str] = &[
    "youtube.com",
    "youtu.be",
    "stackoverflow.com",
    "stackexchange.com",
    "reddit.com",
    "quora.com",
    "medium.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "linkedin.com",
];
pub const MAX_LINKED_URLS_TO_FOLLOW: usize = 4;

pub const DEEP_SCRAPE_CHUNK_SIZE: usize = 250;
pub const DEEP_SCRAPE_TOP_K_MULTIPLIER: usize = 3;
pub const DEEP_SCRAPE_MIN_RELEVANCE_THRESHOLD: f64 = 0.2;

pub const MAX_DEEP_SCRAPE_LITE_RESULTS: usize = 8;
pub const DEEP_SCRAPE_BACK_OFF_BASE_MS: u64 = 300;
pub const DEEP_SCRAPE_ASPECT_ADVANTAGE: &str = "advantage";
pub const DEEP_SCRAPE_ASPECT_BENEFIT: &str = "benefit";
pub const DEEP_SCRAPE_ASPECT_DISADVANTAGE: &str = "disadvantage";
pub const DEEP_SCRAPE_LITE_QUERY_REGION: &str = "wt-wt";
pub const DEEP_SEARCH_URL_FOOTER_HEADER: &str = "\n\nFor more information, see:\n";
pub const DEEP_SEARCH_URL_FOOTER_BULLET: &str = "  • ";

/// Maximum number of retries when DDG Lite search returns an error.
pub const DEEP_SCRAPE_MAX_RETRIES: u32 = 2;

/// Base delay (ms) for exponential back-off between retries.
pub const DEEP_SCRAPE_RETRY_BASE_MS: u64 = 500;

/// Prefix for DDG internal entity pages (e.g. `https://duckduckgo.com/Rust_(programming_language)`).
pub const DEEP_SCRAPE_DDG_ENTITY_PREFIX: &str = "https://duckduckgo.com/";

/// Wikipedia article base URL for resolving DDG entity pages.
pub const DEEP_SCRAPE_WIKIPEDIA_PREFIX: &str = "https://en.wikipedia.org/wiki/";

/// Suffix appended to subject for Wikipedia program fallback URL.
pub const WIKIPEDIA_FALLBACK_PROGRAM_SUFFIX: &str = "_program";

/// Suffix appended to subject for Wikipedia example fallback URL.
pub const WIKIPEDIA_FALLBACK_EXAMPLE_SUFFIX: &str = "_example";

/// Prefix prepended to single-word subject for numerical Wikipedia article.
pub const WIKIPEDIA_FALLBACK_NUMERICAL_PREFIX: &str = "Numerical_";

/// Suffix appended to single-word subject for Wikipedia algorithm article.
pub const WIKIPEDIA_FALLBACK_ALGORITHM_SUFFIX: &str = "_algorithm";

/// Suffix appended to single-word subject for Wikipedia function article.
pub const WIKIPEDIA_FALLBACK_FUNCTION_SUFFIX: &str = "_function";

/// Suffix appended to single-word subject for Wikipedia method article.
pub const WIKIPEDIA_FALLBACK_METHOD_SUFFIX: &str = "_method";

/// Suffix appended to single-word subject for Wikipedia computation article.
pub const WIKIPEDIA_FALLBACK_COMPUTATION_SUFFIX: &str = "_computation";

/// OpenSearch query template for subject program.
pub const OPENSEARCH_QUERY_PROGRAM_TEMPLATE: &str = "{} program";

/// OpenSearch query template for subject program in language.
pub const OPENSEARCH_QUERY_PROGRAM_LANG_TEMPLATE: &str = "{} program python";

/// OpenSearch query template for subject algorithm in language.
pub const OPENSEARCH_QUERY_ALGORITHM_LANG_TEMPLATE: &str = "{} algorithm python";

/// OpenSearch query template for subject implementation.
pub const OPENSEARCH_QUERY_IMPLEMENTATION_TEMPLATE: &str = "{} implementation";

/// Wikipedia OpenSearch API endpoint for resolving article titles from a subject query.
pub const WIKIPEDIA_OPENSEARCH_API: &str =
    "https://en.wikipedia.org/w/api.php?action=opensearch&limit=5&format=json&search=";

/// Protocol-relative URL prefix (`//`).
pub const DEEP_SCRAPE_PROTOCOL_RELATIVE_PREFIX: &str = "//";

/// HTTPS scheme prefix for prepending to protocol-relative URLs.
pub const DEEP_SCRAPE_PROTOCOL_HTTPS: &str = "https:";

/// Absolute redirect path prefix (without host) for DDG redirects.
pub const DEEP_SCRAPE_ABS_REDIRECT_PREFIX: &str = "/l/?uddg=";

/// DDG base URL for constructing absolute URLs from relative paths.
pub const DEEP_SCRAPE_DDG_BASE: &str = "https://duckduckgo.com";

/// Query template for factual/definitional deep search.
pub const DEEP_QUERY_WHAT_IS: &str = "what is ";

/// Query template for definitional deep search.
pub const DEEP_QUERY_DEFINITION: &str = " definition";

/// Query template for causal deep search.
pub const DEEP_QUERY_WHY: &str = "why ";

/// Query template for causal deep search.
pub const DEEP_QUERY_CAUSE: &str = " cause";

/// Query template for causal deep search.
pub const DEEP_QUERY_REASON: &str = " reason";

/// Query template for procedural deep search.
pub const DEEP_QUERY_HOW_TO: &str = "how to ";

/// Query template for procedural deep search.
pub const DEEP_QUERY_TUTORIAL: &str = " tutorial";

/// Query template for procedural deep search.
pub const DEEP_QUERY_GUIDE: &str = " guide";

/// Query template for evaluative deep search.
pub const DEEP_QUERY_ADVANTAGES: &str = "advantages of ";

/// Query template for evaluative deep search.
pub const DEEP_QUERY_BENEFITS: &str = "benefits of ";

/// Query template for evaluative deep search.
pub const DEEP_QUERY_FEATURES: &str = " features";

/// Query template for evaluative deep search.
pub const DEEP_QUERY_WHY_USE: &str = "why use ";

/// Query template for evaluative deep search.
pub const DEEP_QUERY_PROS_CONS: &str = " pros and cons";

/// Query template for analogical deep search.
pub const DEEP_QUERY_VS: &str = " vs ";

/// Query template for analogical deep search.
pub const DEEP_QUERY_COMPARE: &str = "compare ";

/// Query template for analogical deep search.
pub const DEEP_QUERY_DIFFERENCES: &str = "differences between ";

/// Query template for generic deep search.
pub const DEEP_QUERY_OVERVIEW: &str = " overview";

/// Query template for generic deep search.
pub const DEEP_QUERY_EXPLAINED: &str = " explained";

/// Query suffix for algorithm-focused deep search (task-execution mode).
pub const DEEP_QUERY_ALGORITHM: &str = " algorithm";

/// Query suffix for recursive-implementation deep search (task-execution mode).
pub const DEEP_QUERY_RECURSIVE: &str = " recursive";

/// Query suffix for iterative-implementation deep search (task-execution mode).
pub const DEEP_QUERY_ITERATIVE: &str = " iterative";

/// Query suffix for code-example deep search (task-execution mode).
pub const DEEP_QUERY_EXAMPLE: &str = " example";

/// Query suffix targeting code implementations in deep search.
pub const DEEP_QUERY_CODE_IMPL: &str = " code implementation";

/// Query suffix for how-to-implement style deep search (task-execution mode).
pub const DEEP_QUERY_HOW_TO_IMPLEMENT: &str = "how to implement ";

/// Query suffix for code snippet deep search (task-execution mode).
pub const DEEP_QUERY_CODE_SNIPPET: &str = " code snippet";

/// Query prefix for "compute X in language" style code-finding queries.
pub const DEEP_QUERY_COMPUTE_IN: &str = "compute";

/// Query suffix for "X program" style code-finding queries.
pub const DEEP_QUERY_PROGRAM: &str = " program";

/// Query suffix for "X function" style code-finding queries.
pub const DEEP_QUERY_FUNCTION: &str = " function";

/// Query suffix for "X tutorial" style code-finding queries (reuses existing DEEP_QUERY_TUTORIAL).
pub const DEEP_QUERY_CODE_TUTORIAL: &str = " tutorial";

/// Site restriction fragment targeting StackOverflow for code queries.
///
/// NOTE: DDG Lite does not support `site:` operators reliably.
/// These constants are retained for reference but are NOT used in query construction.
pub const DEEP_QUERY_SITE_STACKOVERFLOW: &str = "";

/// Site restriction fragment targeting GeeksForGeeks for code queries.
pub const DEEP_QUERY_SITE_GFG: &str = "";

/// Site restriction fragment targeting GitHub for code queries.
pub const DEEP_QUERY_SITE_GITHUB: &str = "";

/// Site restriction fragment targeting programiz.com for code queries.
pub const DEEP_QUERY_SITE_PROGRAMIZ: &str = "";

/// Query suffix for thematic poetry deep search (task-execution creative mode).
pub const DEEP_QUERY_THEMES: &str = " themes";

/// Query suffix for meaning-focused deep search (task-execution creative mode).
pub const DEEP_QUERY_MEANING: &str = " meaning";

/// Query suffix for philosophy-focused deep search (task-execution creative mode).
pub const DEEP_QUERY_PHILOSOPHY: &str = " philosophy";

/// Query suffix for metaphor-focused deep search (task-execution creative mode).
pub const DEEP_QUERY_METAPHORS: &str = " metaphors";

/// Query suffix for human-experience deep search (task-execution creative mode).
pub const DEEP_QUERY_EXPERIENCE: &str = " human experience";

/// URL path fragments that indicate a non-article page (sitemaps, feeds,
/// login pages, app-store links, navigation pages, etc.).
///
/// Any scraped URL whose lowercase form contains one of these fragments is
/// excluded from the TRAM response footer to avoid showing junk links.
pub const URL_REJECT_PATTERNS: &[&str] = &[
    "sitemap",
    "/feed",
    "/rss",
    ".xml",
    "/tag/",
    "/category/",
    "/author/",
    "/page/",
    "/search?",
    "/search/",
    "/login",
    "/signup",
    "/register",
    "play.google.com",
    "apps.apple.com",
    "itunes.apple.com",
    "/privacy",
    "/terms",
    "/cookie",
    "/about-us",
    "/about/",
    "/contact",
    "/wp-admin",
    "/wp-login",
    "/cdn-cgi/",
    "javascript:",
    "mailto:",
    "#",
    "/alzheimer",
    "/dementia",
    "/anxiety",
    "/depression",
    "/sweepstake",
    "magazine",
    "subscription",
    "/chembl",
    "/chebi",
    "jmol",
    "browsehappy.com",
    "cas_rn=",
    "utm_source=",
    "utm_medium=",
    "utm_campaign=",
    "utm_content=",
    "utm_term=",
    "/careers",
    "/jobs",
    "/press",
    "/advertise",
    "/newsletter",
    "/unsubscribe",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",
    "pinterest.com",
    "/share?",
    "/print/",
    "/amp/",
    "/wiki/What_Are",
    "/wiki/What_Is",
    "/wiki/Show_Me",
    "/wiki/How_To",
];

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Well-known programming language names used to disambiguate evaluative WASM
/// queries (e.g. "advantages of rust" → "advantages of rust programming
/// language") so DuckDuckGo returns the PL article rather than the material.
pub const KNOWN_PROGRAMMING_LANGUAGES: &[&str] = &[
    "rust",
    "python",
    "javascript",
    "typescript",
    "go",
    "java",
    "kotlin",
    "swift",
    "ruby",
    "scala",
    "haskell",
    "c",
    "dart",
    "elixir",
    "clojure",
    "erlang",
    "julia",
    "nim",
    "zig",
    "lua",
    "perl",
    "r",
    "php",
    "bash",
    "shell",
    "groovy",
    "ocaml",
    "fsharp",
    "crystal",
    "purescript",
    "elm",
    "gleam",
];

/// Substring that indicates a topic already includes a PL disambiguator,
/// preventing double-appending (e.g. "Rust Programming Language programming language").
pub const PL_DISAMBIGUATION_SUFFIX: &str = "programming language";

/// Wikipedia parenthetical disambiguation suffix for programming-language articles
/// (e.g. "Rust (programming language)").
pub const WIKIPEDIA_PL_DISAMBIG: &str = " (programming language)";

/// DDG/Wikipedia query prefix for code-example searches: `"{language} {subject} example"`.
pub const WASM_CODE_EXAMPLE_QUERY: &str = " example";

/// DDG query prefix for how-to code searches: `"how to write {subject} in {language}"`.
pub const WASM_HOW_TO_WRITE_PREFIX: &str = "how to write ";

/// DDG query connector used between subject and language in how-to queries.
pub const WASM_HOW_TO_IN_CONNECTOR: &str = " in ";

/// DDG query template prefix for creative/poem topic searches.
pub const WASM_POEM_ABOUT_PREFIX: &str = "poem about ";

/// DDG query template prefix for famous-poem topic searches.
pub const WASM_FAMOUS_POEMS_PREFIX: &str = "famous poems about ";

/// Wikipedia fetch prefix emitted to `fetch_queries` for traceability.
pub const WIKIPEDIA_FETCH_QUERY_PREFIX: &str = "wikipedia:";

/// Wikipedia OpenSearch API endpoint (CORS-safe, `origin=*`).
pub const WIKIPEDIA_OPENSEARCH_URL: &str =
    "https://en.wikipedia.org/w/api.php?action=opensearch&limit=5&format=json&origin=*&search=";

/// Wikipedia mobile-sections REST API base URL.
pub const WIKIPEDIA_MOBILE_SECTIONS_URL: &str =
    "https://en.wikipedia.org/api/rest_v1/page/mobile-sections/";

/// Knowledge-index source label for the primary Wikipedia article corpus.
pub const WIKIPEDIA_SOURCE: &str = "wikipedia";

/// Knowledge-index source label for alternate Wikipedia code-example corpus.
pub const WIKIPEDIA_ALT_SOURCE: &str = "wikipedia-alt";

/// Knowledge-index source label for per-task Wikipedia query fallback.
pub const WIKIPEDIA_QUERY_SOURCE: &str = "wikipedia-query";

/// Minimum number of characters a `<pre>` / `<code>` block must have to be
/// kept as a code snippet. Shorter blocks are noise (XML tags, one-liners).
pub const MIN_CODE_SNIPPET_CHARS: usize = 10;

/// Canonical Hello-World source snippets indexed by lowercase language name.
///
/// Used in WASM builds as a local fallback when Wikipedia code extraction
/// yields no code blocks, ensuring `CodeResponder` always has a valid snippet
/// to present for the most common programming languages.
pub const CANONICAL_HELLO_WORLD: &[(&str, &str)] = &[
    ("python", "print(\"Hello, World!\")"),
    ("javascript", "console.log(\"Hello, World!\");"),
    ("typescript", "console.log(\"Hello, World!\");"),
    ("rust", "fn main() {\n    println!(\"Hello, World!\");\n}"),
    (
        "go",
        "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello, World!\")\n}",
    ),
    (
        "java",
        "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}",
    ),
    (
        "c",
        "#include <stdio.h>\n\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}",
    ),
    (
        "c++",
        "#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}",
    ),
    (
        "c#",
        "using System;\n\nclass Program {\n    static void Main() {\n        Console.WriteLine(\"Hello, World!\");\n    }\n}",
    ),
    ("ruby", "puts \"Hello, World!\""),
    ("kotlin", "fun main() {\n    println(\"Hello, World!\")\n}"),
    ("swift", "print(\"Hello, World!\")"),
    ("php", "<?php\necho \"Hello, World!\\n\";"),
    ("scala", "@main def hello() = println(\"Hello, World!\")"),
    (
        "haskell",
        "main :: IO ()\nmain = putStrLn \"Hello, World!\"",
    ),
    ("lua", "print(\"Hello, World!\")"),
    ("perl", "print \"Hello, World!\\n\";"),
    ("r", "cat(\"Hello, World!\\n\")"),
    ("dart", "void main() {\n  print('Hello, World!');\n}"),
    ("elixir", "IO.puts \"Hello, World!\""),
    ("clojure", "(println \"Hello, World!\")"),
    (
        "erlang",
        "-module(hello).\n-export([start/0]).\nstart() -> io:fwrite(\"Hello, World!\\n\").",
    ),
    ("julia", "println(\"Hello, World!\")"),
    ("nim", "echo \"Hello, World!\""),
    (
        "zig",
        "const std = @import(\"std\");\npub fn main() !void {\n    const stdout = std.io.getStdOut().writer();\n    try stdout.print(\"Hello, World!\\n\", .{});\n}",
    ),
    ("bash", "echo \"Hello, World!\""),
    ("shell", "echo \"Hello, World!\""),
];
