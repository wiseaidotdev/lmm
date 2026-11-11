// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::code_responder::CodeResponder;

#[test]
fn generate_python_hello_world_contains_print() {
    let code = CodeResponder::generate("python", "hello world", "");
    assert!(
        code.contains("print"),
        "Python hello world should use print(), got: {code}"
    );
    assert!(code.contains("Hello"), "Should contain 'Hello'");
}

#[test]
fn generate_rust_hello_world_contains_println() {
    let code = CodeResponder::generate("rust", "hello world", "");
    assert!(
        code.contains("println!"),
        "Rust hello world should use println!, got: {code}"
    );
    assert!(code.contains("fn main"), "Should contain fn main");
}

#[test]
fn generate_javascript_hello_world_contains_console_log() {
    let code = CodeResponder::generate("javascript", "hello world", "");
    assert!(
        code.contains("console.log"),
        "JS hello world should use console.log, got: {code}"
    );
}

#[test]
fn generate_go_hello_world_contains_fmt_println() {
    let code = CodeResponder::generate("go", "hello world", "");
    assert!(
        code.contains("fmt.Println"),
        "Go hello world should use fmt.Println, got: {code}"
    );
    assert!(code.contains("package main"), "Should contain package main");
}

#[test]
fn generate_java_hello_world_contains_system_out() {
    let code = CodeResponder::generate("java", "hello world", "");
    assert!(
        code.contains("System.out.println"),
        "Java should use System.out.println, got: {code}"
    );
}

#[test]
fn generate_cpp_hello_world_contains_cout() {
    let code = CodeResponder::generate("cpp", "hello world", "");
    assert!(
        code.contains("cout") || code.contains("std::cout"),
        "C++ should use cout, got: {code}"
    );
}

#[test]
fn generate_ruby_hello_world_contains_puts() {
    let code = CodeResponder::generate("ruby", "hello world", "");
    assert!(
        code.contains("puts"),
        "Ruby hello world should use puts, got: {code}"
    );
}

#[test]
fn generate_python_fibonacci_contains_def_fibonacci() {
    let code = CodeResponder::generate("python", "fibonacci sequence", "");
    assert!(
        code.contains("fibonacci"),
        "Should define fibonacci function, got: {code}"
    );
}

#[test]
fn generate_rust_fibonacci_contains_fn_fibonacci() {
    let code = CodeResponder::generate("rust", "fibonacci numbers", "");
    assert!(
        code.contains("fn fibonacci"),
        "Should define fn fibonacci, got: {code}"
    );
}

#[test]
fn generate_python_factorial_contains_factorial() {
    let code = CodeResponder::generate("python", "factorial", "");
    assert!(
        code.contains("factorial"),
        "Should define factorial, got: {code}"
    );
}

#[test]
fn format_code_block_wraps_in_backtick_fence() {
    let block = CodeResponder::format_code_block("python", "print('hello')");
    assert!(
        block.starts_with("```python"),
        "Should start with ```python"
    );
    assert!(block.ends_with("```"), "Should end with ```");
    assert!(block.contains("print('hello')"), "Should contain the code");
}

#[test]
fn format_code_block_trims_code_whitespace() {
    let block = CodeResponder::format_code_block("rust", "  fn main() {}  ");
    assert!(
        block.contains("fn main() {}"),
        "Should trim inner whitespace"
    );
}

#[test]
fn format_code_block_produces_correct_exact_string() {
    let block = CodeResponder::format_code_block("rust", "fn main() {}");
    assert_eq!(block, "```rust\nfn main() {}\n```");
}

#[test]
fn detect_task_type_hello_world() {
    assert_eq!(
        CodeResponder::detect_task_type("hello world program"),
        "hello_world"
    );
    assert_eq!(
        CodeResponder::detect_task_type("write hello"),
        "hello_world"
    );
}

#[test]
fn detect_task_type_fibonacci() {
    assert_eq!(
        CodeResponder::detect_task_type("fibonacci sequence"),
        "fibonacci"
    );
    assert_eq!(
        CodeResponder::detect_task_type("compute fib sequence"),
        "fibonacci"
    );
}

#[test]
fn detect_task_type_factorial() {
    assert_eq!(
        CodeResponder::detect_task_type("factorial function"),
        "factorial"
    );
}

#[test]
fn detect_task_type_sort() {
    assert_eq!(CodeResponder::detect_task_type("sort an array"), "sort");
    assert_eq!(
        CodeResponder::detect_task_type("bubble sort implementation"),
        "sort"
    );
}

#[test]
fn detect_task_type_read_file() {
    assert_eq!(
        CodeResponder::detect_task_type("read file contents"),
        "read_file"
    );
    assert_eq!(
        CodeResponder::detect_task_type("open file and parse"),
        "read_file"
    );
}

#[test]
fn detect_task_type_http_request() {
    assert_eq!(
        CodeResponder::detect_task_type("make http request"),
        "http_request"
    );
    assert_eq!(
        CodeResponder::detect_task_type("fetch data from api"),
        "http_request"
    );
}

#[test]
fn detect_task_type_generic_fallback() {
    assert_eq!(
        CodeResponder::detect_task_type("unknown algorithm"),
        "generic"
    );
    assert_eq!(CodeResponder::detect_task_type("do something"), "generic");
}

#[test]
fn generate_with_empty_corpus_falls_back_to_template() {
    let code = CodeResponder::generate("python", "hello world", "");
    assert!(!code.is_empty(), "Should return template for empty corpus");
}

#[test]
fn generate_unknown_language_hello_world_fallback() {
    let code = CodeResponder::generate("brainfuck", "hello world", "");
    assert!(
        !code.is_empty(),
        "Should return something even for unknown language"
    );
}

#[test]
fn generate_known_language_generic_task_returns_non_empty() {
    let code = CodeResponder::generate("rust", "binary search", "");
    assert!(
        !code.is_empty(),
        "Generic task should return non-empty code"
    );
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
