// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Code Responder
//!
//! Generates code snippets for common task types when the pipeline operates
//! in [`crate::cognition::thought::classifier::ReasoningMode::TaskExecution`] mode.
//!
//! Code is first extracted from the ingested `kb_corpus`; when no relevant
//! snippet is found the module falls back to language-specific built-in
//! templates for well-known tasks (hello world, fibonacci, etc.).
//!
//! ## Examples
//!
//! ```rust
//! use lmm::cognition::code_responder::CodeResponder;
//!
//! let code = CodeResponder::generate("python", "hello world", "");
//! assert!(code.contains("print"));
//! ```

/// Generates and formats code snippets for task-execution prompts.
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
    /// Generates a code snippet for `task` in `language`, optionally extracting
    /// from `kb_corpus` before falling back to built-in templates.
    ///
    /// # Arguments
    ///
    /// * `language`  - Target language identifier (e.g. `"python"`, `"rust"`).
    /// * `task`      - Task description from the user prompt.
    /// * `kb_corpus` - Ingested knowledge text; searched for code snippets first.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::code_responder::CodeResponder;
    ///
    /// let code = CodeResponder::generate("rust", "hello world", "");
    /// assert!(code.contains("println!") || code.contains("fn main"));
    /// ```
    pub fn generate(language: &str, task: &str, kb_corpus: &str) -> String {
        if let Some(extracted) = Self::extract_from_corpus(kb_corpus, language) {
            return extracted;
        }
        let task_type = Self::detect_task_type(task);
        Self::template_for(language, task_type)
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

    /// Detects the high-level task type from a prompt string.
    ///
    /// Returns one of: `"hello_world"`, `"fibonacci"`, `"factorial"`,
    /// `"sort"`, `"read_file"`, `"http_request"`, or `"generic"`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm::cognition::code_responder::CodeResponder;
    ///
    /// assert_eq!(CodeResponder::detect_task_type("hello world program"), "hello_world");
    /// assert_eq!(CodeResponder::detect_task_type("fibonacci sequence"), "fibonacci");
    /// assert_eq!(CodeResponder::detect_task_type("unknown task"), "generic");
    /// ```
    pub fn detect_task_type(prompt: &str) -> &'static str {
        let lower = prompt.to_ascii_lowercase();
        if lower.contains("hello") || lower.contains("hi world") {
            "hello_world"
        } else if lower.contains("fibonacci") || lower.contains("fib sequence") {
            "fibonacci"
        } else if lower.contains("factorial") {
            "factorial"
        } else if lower.contains("sort") {
            "sort"
        } else if lower.contains("read file") || lower.contains("open file") {
            "read_file"
        } else if lower.contains("http") || lower.contains("request") || lower.contains("fetch") {
            "http_request"
        } else {
            "generic"
        }
    }

    /// Attempts to extract a code snippet from `corpus` for `language`.
    ///
    /// Looks for lines containing language-specific markers and code
    /// punctuation (`{`, `(`, `;`).  Returns `None` when fewer than
    /// two matching lines are found.
    ///
    /// # Time Complexity
    ///
    /// O(|corpus|)
    fn extract_from_corpus(corpus: &str, language: &str) -> Option<String> {
        if corpus.is_empty() {
            return None;
        }
        let markers = lang_code_markers(language);
        let code_lines: Vec<&str> = corpus
            .lines()
            .filter(|line| {
                let l = line.trim();
                !l.is_empty()
                    && markers.iter().any(|m| l.contains(m))
                    && (l.contains('{') || l.contains('(') || l.contains(';') || l.contains("->"))
            })
            .take(15)
            .collect();
        if code_lines.len() < 2 {
            return None;
        }
        Some(code_lines.join("\n"))
    }

    /// Returns a built-in template for `language` and `task_type`.
    ///
    /// Falls back to a generic comment when the combination is not templated.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    fn template_for(language: &str, task_type: &str) -> String {
        match (language, task_type) {
            ("python", "hello_world") => r#"print("Hello, World!")"#.to_string(),
            ("python", "fibonacci") => concat!(
                "def fibonacci(n):\n",
                "    if n <= 1:\n",
                "        return n\n",
                "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n",
                "print(fibonacci(10))"
            )
            .to_string(),
            ("python", "factorial") => concat!(
                "def factorial(n):\n",
                "    if n == 0:\n",
                "        return 1\n",
                "    return n * factorial(n - 1)\n\n",
                "print(factorial(5))"
            )
            .to_string(),
            ("python", "sort") => concat!(
                "numbers = [64, 34, 25, 12, 22, 11, 90]\n",
                "numbers.sort()\n",
                "print(numbers)"
            )
            .to_string(),
            ("rust", "hello_world") => {
                concat!("fn main() {\n", "    println!(\"Hello, World!\");\n", "}").to_string()
            }
            ("rust", "fibonacci") => concat!(
                "fn fibonacci(n: u64) -> u64 {\n",
                "    if n <= 1 { return n; }\n",
                "    fibonacci(n - 1) + fibonacci(n - 2)\n",
                "}\n\n",
                "fn main() {\n",
                "    println!(\"{}\", fibonacci(10));\n",
                "}"
            )
            .to_string(),
            ("rust", "factorial") => concat!(
                "fn factorial(n: u64) -> u64 {\n",
                "    if n == 0 { 1 } else { n * factorial(n - 1) }\n",
                "}\n\n",
                "fn main() {\n",
                "    println!(\"{}\", factorial(5));\n",
                "}"
            )
            .to_string(),
            ("rust", "sort") => concat!(
                "fn main() {\n",
                "    let mut numbers = vec![64, 34, 25, 12, 22, 11, 90];\n",
                "    numbers.sort();\n",
                "    println!(\"{:?}\", numbers);\n",
                "}"
            )
            .to_string(),
            ("javascript", "hello_world") | ("typescript", "hello_world") => {
                r#"console.log("Hello, World!");"#.to_string()
            }
            ("javascript", "fibonacci") => concat!(
                "function fibonacci(n) {\n",
                "    if (n <= 1) return n;\n",
                "    return fibonacci(n - 1) + fibonacci(n - 2);\n",
                "}\n",
                "console.log(fibonacci(10));"
            )
            .to_string(),
            ("go", "hello_world") => concat!(
                "package main\n\n",
                "import \"fmt\"\n\n",
                "func main() {\n",
                "    fmt.Println(\"Hello, World!\")\n",
                "}"
            )
            .to_string(),
            ("java", "hello_world") => concat!(
                "public class HelloWorld {\n",
                "    public static void main(String[] args) {\n",
                "        System.out.println(\"Hello, World!\");\n",
                "    }\n",
                "}"
            )
            .to_string(),
            ("cpp", "hello_world") => concat!(
                "#include <iostream>\n\n",
                "int main() {\n",
                "    std::cout << \"Hello, World!\" << std::endl;\n",
                "    return 0;\n",
                "}"
            )
            .to_string(),
            ("ruby", "hello_world") => r#"puts "Hello, World!""#.to_string(),
            ("swift", "hello_world") => r#"print("Hello, World!")"#.to_string(),
            ("kotlin", "hello_world") => r#"fun main() { println("Hello, World!") }"#.to_string(),
            ("bash", "hello_world") | ("shell", "hello_world") => {
                r#"echo "Hello, World!""#.to_string()
            }
            (_, "hello_world") => r#"print("Hello, World!")"#.to_string(),
            (lang, _) => format!(
                "// {}, no built-in template for this task type.\n// Refer to the {} documentation.",
                lang, lang
            ),
        }
    }
}

/// Returns language-specific code marker strings for corpus extraction.
///
/// # Time Complexity
///
/// O(1)
fn lang_code_markers(language: &str) -> &'static [&'static str] {
    match language {
        "python" => &["def ", "print(", "import ", "class ", "return ", "lambda "],
        "rust" => &[
            "fn ", "let ", "println!", "struct ", "impl ", "pub ", "use ",
        ],
        "javascript" | "typescript" => {
            &["function ", "const ", "let ", "var ", "console.log", "=>"]
        }
        "go" => &["func ", "fmt.", "package ", "import "],
        "java" => &["public ", "class ", "void ", "System.out"],
        "cpp" => &["#include", "std::", "cout", "int main"],
        "ruby" => &["def ", "puts ", "end", "require "],
        _ => &["function", "print", "main", "return"],
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
