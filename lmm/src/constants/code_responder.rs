// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Code Responder Constants

/// Minimum number of code-like lines required to treat corpus content as a usable snippet.
pub const MIN_CODE_LINES: usize = 2;

/// Maximum number of lines taken from the corpus when extracting a code snippet.
pub const MAX_CODE_LINES_TO_TAKE: usize = 30;

/// Minimum number of characters the KB corpus must contain before code extraction is attempted.
///
/// Below this threshold the topic is considered under-researched and the user is
/// prompted to supply a more specific query.
pub const MIN_CORPUS_CHARS_FOR_CODE: usize = 100;

/// Template for the prompt-improvement suggestion shown when no code can be extracted.
///
/// `{subject}` is replaced with the detected task subject and `{language}` with the
/// target language name.
pub const CODE_IMPROVEMENT_HINT: &str = "\
No code could be generated for \"{subject}\" in {language} from the available \
knowledge base.

To get a working code response, try a more specific query such as:\n\
  • write a {language} program for {subject} using the recursive approach\n\
  • write a {language} program for {subject} step by step with an example\n\
  • show me a {subject} algorithm example in {language}\n\n\
Adding context about the approach (recursive, iterative, formula-based) or the \
domain (mathematics, data processing, file I/O) helps the LMM find and \
assemble accurate code from its knowledge base.";

/// Prefix prepended to code-block content when ingesting it into the knowledge index.
///
/// The scraper wraps every extracted `<pre>`, `<code>`, or fenced block with this
/// prefix so `extract_from_corpus` can locate verbatim code regions quickly.
pub const CODE_BLOCK_MARKER: &str = "CODEBLOCK::";

/// Minimum number of non-whitespace characters a code block must have to be ingested.
pub const MIN_CODE_BLOCK_CHARS: usize = 20;

/// Maximum number of characters extracted from a single code block.
pub const MAX_CODE_BLOCK_CHARS: usize = 4096;

/// Number of leading spaces that identify an indented code line in plain text.
pub const INDENTED_CODE_SPACES: usize = 4;

/// Minimum fraction of lines in a candidate block that must look like code
/// (contain at least one code-punctuation character) to accept the block.
pub const MIN_CODE_LINE_RATIO: f64 = 0.4;

/// Characters whose presence in a line strongly suggests it is code rather than prose.
pub const CODE_PUNCT_CHARS: &[char] = &['(', ')', '{', '}', '[', ']', '=', ';', ':', '-', '>'];

/// Minimum number of language-keyword hits a `CODEBLOCK::` segment must score
/// to be accepted by `extract_marked_blocks`.  Segments scoring below this
/// threshold are math prose or pseudocode with no recognisable programming
/// constructs and are silently dropped.
pub const MIN_MARKED_BLOCK_SCORE: usize = 1;

/// Multiplier applied to code punctuation count when checking for mathematical prose.
pub const MATH_PROSE_RATIO_MULTIPLIER: usize = 2;

/// Minimum character length required for a subject token when scoring relevance.
pub const SUBJECT_TOKEN_MIN_LEN: usize = 3;

/// Weight factor applied to subject token occurrences during candidate block scoring.
pub const SUBJECT_RELEVANCE_BOOST_FACTOR: f64 = 2.0;

/// Fallback language tag used when comparing fenced block language tags.
pub const FALLBACK_LANGUAGE_TAG: &str = "text";

/// Introduction phrase preceding a generated code response.
pub const CODE_INTRO_TEMPLATE: &str = "Here is a {language} program for {subject}:";

/// Placeholder identifier for subject substitution.
pub const PLACEHOLDER_SUBJECT: &str = "{subject}";

/// Placeholder identifier for language substitution.
pub const PLACEHOLDER_LANGUAGE: &str = "{language}";

/// Target language identifier for Python.
pub const LANGUAGE_PYTHON: &str = "python";

/// Target language identifier for Rust.
pub const LANGUAGE_RUST: &str = "rust";

/// Target language identifier for JavaScript.
pub const LANGUAGE_JAVASCRIPT: &str = "javascript";

/// Target language identifier for TypeScript.
pub const LANGUAGE_TYPESCRIPT: &str = "typescript";

/// Target language identifier for Go.
pub const LANGUAGE_GO: &str = "go";

/// Target language identifier for Java.
pub const LANGUAGE_JAVA: &str = "java";

/// Target language identifier for C++.
pub const LANGUAGE_CPP: &str = "cpp";

/// Target language identifier for Ruby.
pub const LANGUAGE_RUBY: &str = "ruby";

/// Code-line start markers recognized for Python.
pub const MARKER_PYTHON: &[&str] = &[
    "def ", "print(", "import ", "class ", "return ", "lambda ", "for ", "if ",
];

/// Code-line start markers recognized for Rust.
pub const MARKER_RUST: &[&str] = &[
    "fn ", "let ", "println!", "struct ", "impl ", "pub ", "use ", "mut ",
];

/// Code-line start markers recognized for JavaScript/TypeScript.
pub const MARKER_JAVASCRIPT: &[&str] = &[
    "function ",
    "const ",
    "let ",
    "var ",
    "console.log",
    "=>",
    "return ",
];

/// Code-line start markers recognized for Go.
pub const MARKER_GO: &[&str] = &["func ", "fmt.", "package ", "import ", "var ", ":="];

/// Code-line start markers recognized for Java.
pub const MARKER_JAVA: &[&str] = &[
    "public ",
    "class ",
    "void ",
    "System.out",
    "return ",
    "int ",
    "String ",
];

/// Code-line start markers recognized for C++.
pub const MARKER_CPP: &[&str] = &["#include", "std::", "cout", "int main", "void ", "return "];

/// Code-line start markers recognized for Ruby.
pub const MARKER_RUBY: &[&str] = &["def ", "puts ", "end", "require ", "return ", "each"];

/// Default code-line start markers recognized when language is unlisted.
pub const MARKER_DEFAULT: &[&str] = &["function", "print", "main", "return", "def "];

/// Task-type label returned when the prompt asks for a hello-world program.
pub const TASK_TYPE_HELLO_WORLD: &str = "hello_world";

/// Task-type label returned when the prompt asks for a Fibonacci implementation.
pub const TASK_TYPE_FIBONACCI: &str = "fibonacci";

/// Task-type label returned when the prompt asks for a factorial implementation.
pub const TASK_TYPE_FACTORIAL: &str = "factorial";

/// Task-type label returned when the prompt involves sorting.
pub const TASK_TYPE_SORT: &str = "sort";

/// Task-type label returned when the prompt asks for file I/O.
pub const TASK_TYPE_READ_FILE: &str = "read_file";

/// Task-type label returned when the prompt asks for HTTP networking.
pub const TASK_TYPE_HTTP_REQUEST: &str = "http_request";

/// Task-type label returned when no specific category is detected.
pub const TASK_TYPE_GENERIC: &str = "generic";

/// Built-in hello-world template for Python.
pub const TEMPLATE_PYTHON_HELLO: &str = "print(\"Hello, World!\")";

/// Built-in hello-world template for Rust.
pub const TEMPLATE_RUST_HELLO: &str = "fn main() {\n    println!(\"Hello, World!\");\n}";

/// Built-in hello-world template for JavaScript.
pub const TEMPLATE_JS_HELLO: &str = "console.log(\"Hello, World!\");";

/// Built-in hello-world template for Go.
pub const TEMPLATE_GO_HELLO: &str =
    "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello, World!\")\n}";

/// Built-in hello-world template for Java.
pub const TEMPLATE_JAVA_HELLO: &str = "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}";

/// Built-in hello-world template for C++.
pub const TEMPLATE_CPP_HELLO: &str = "#include <iostream>\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}";

/// Built-in hello-world template for Ruby.
pub const TEMPLATE_RUBY_HELLO: &str = "puts \"Hello, World!\"";

/// Built-in Fibonacci template for Python.
pub const TEMPLATE_PYTHON_FIBONACCI: &str = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)";

/// Built-in Fibonacci template for Rust.
pub const TEMPLATE_RUST_FIBONACCI: &str = "fn fibonacci(n: u64) -> u64 {\n    if n <= 1 { return n; }\n    fibonacci(n - 1) + fibonacci(n - 2)\n}";

/// Built-in factorial template for Python.
pub const TEMPLATE_PYTHON_FACTORIAL: &str =
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)";

/// Generic fallback template used when no better template matches.
pub const TEMPLATE_GENERIC: &str = "// TODO: implement the requested functionality";

/// Unicode characters associated with mathematical notation (not programming).
///
/// A `CODEBLOCK::` segment whose non-ASCII characters are drawn predominantly
/// from this set and that scores 0 on code-language markers is classified as
/// math prose and rejected.
pub const MATH_UNICODE_CHARS: &[char] = &[
    '′', '″', '∫', '∂', '∑', '∏', '∞', '≤', '≥', '≠', '≈', '∈', '∉', '⊂', '⊃', '⊆', '⊇', '∪', '∩',
    '∧', '∨', '¬', '→', '↔', '⇒', '⇔', '∀', '∃', '∅', '√', '∛', '∜', '±', '×', '÷', '·', 'α', 'β',
    'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'λ', 'μ', 'π', 'σ', 'τ', 'φ', 'ψ', 'ω',
];

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
