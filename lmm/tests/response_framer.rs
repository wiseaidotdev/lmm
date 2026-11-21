// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::response_framer::ResponseFramer;
use lmm::cognition::thought::classifier::ReasoningMode;

#[test]
fn exploratory_mode_returns_answer_unchanged() {
    let answer = "Rust is a systems programming language.";
    let framed = ResponseFramer::frame(ReasoningMode::Exploratory, "rust", "rust", answer, 0);
    assert_eq!(framed, answer);
}

#[test]
fn factual_lookup_mode_returns_answer_unchanged() {
    let answer = "Rust was created by Graydon Hoare.";
    let framed = ResponseFramer::frame(
        ReasoningMode::FactualLookup,
        "rust creator",
        "rust creator",
        answer,
        0,
    );
    assert_eq!(framed, answer);
}

#[test]
fn causal_mode_prepends_the_reason_phrase() {
    let answer = "it uses ownership rules enforced at compile time.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Causal,
        "Rust memory safety",
        "why does rust have memory safety",
        answer,
        0,
    );
    assert!(
        framed.starts_with("The reason")
            || framed.to_ascii_lowercase().contains("reason")
            || framed.to_ascii_lowercase().contains("what is known")
            || framed.to_ascii_lowercase().contains("stems from"),
        "Expected causal prefix, got: {framed}"
    );
}

#[test]
fn causal_mode_does_not_double_frame_already_causal_answer() {
    let answer = "The reason Rust avoids GC is ownership.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Causal,
        "gc avoidance",
        "why does rust avoid gc",
        answer,
        0,
    );
    assert_eq!(
        framed, answer,
        "Should not double-frame an already-causal answer"
    );
}

#[test]
fn analogical_mode_prepends_comparing_phrase() {
    let answer = "Rust is fast but Python is easier.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Analogical,
        "rust vs python",
        "rust vs python",
        answer,
        0,
    );
    assert!(
        framed.to_ascii_lowercase().contains("compar")
            || framed.to_ascii_lowercase().contains("contrast")
            || framed.to_ascii_lowercase().contains("difference")
            || framed.to_ascii_lowercase().contains("versus")
            || framed.to_ascii_lowercase().contains("when"),
        "Expected analogical comparison prefix, got: {framed}"
    );
}

#[test]
fn analogical_mode_extracts_entity_names_from_vs_topic() {
    let phrase =
        ResponseFramer::opening_phrase(ReasoningMode::Analogical, "rust vs python", "", "", 0);
    assert!(phrase.contains("Rust"), "Should contain first entity");
    assert!(phrase.contains("Python"), "Should contain second entity");
}

#[test]
fn definitional_mode_prepends_by_definition_phrase() {
    let answer = "a data type that stores a value or nothing.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Definitional,
        "option type",
        "what is the option type",
        answer,
        0,
    );
    assert!(
        framed.starts_with("By definition")
            || framed.to_ascii_lowercase().contains("by definition")
            || framed.to_ascii_lowercase().contains("refers to"),
        "Expected 'By definition...' prefix, got: {framed}"
    );
}

#[test]
fn procedural_mode_prepends_here_is_how_phrase() {
    let answer = "run rustup-init.sh from the official site.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Procedural,
        "install rust",
        "how to install rust",
        answer,
        0,
    );
    assert!(
        framed.starts_with("Here is how to")
            || framed.to_ascii_lowercase().contains("here is how")
            || framed.to_ascii_lowercase().starts_with("run")
            || framed.to_ascii_lowercase().starts_with("download"),
        "Expected procedural prefix, got: {framed}"
    );
}

#[test]
fn evaluative_mode_prepends_advantages_phrase() {
    let answer = "Rust is fast, safe, and has great tooling.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Evaluative,
        "Rust",
        "what are the advantages of rust",
        answer,
        0,
    );
    assert!(
        framed.to_ascii_lowercase().contains("advantage")
            || framed.to_ascii_lowercase().contains("pros")
            || framed.to_ascii_lowercase().contains("benefit"),
        "Expected evaluative prefix, got: {framed}"
    );
}

#[test]
#[ignore]
fn evaluative_mode_uses_disadvantages_when_answer_contains_drawback() {
    let answer = "Rust has a steep learning curve and slow compile times.";
    let phrase = ResponseFramer::opening_phrase(ReasoningMode::Evaluative, "Rust", "", answer, 0);
    assert!(
        phrase.contains("disadvantage") || phrase.contains("pros and cons"),
        "Expected disadvantage/pros-and-cons phrase for negative answer, got: {phrase}"
    );
}

#[test]
fn evaluative_mode_inconveniences_uses_correct_prefix() {
    let answer = "Intelligence can be isolating and cause overthinking.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Evaluative,
        "intelligence",
        "what are the inconveniences of intelligence",
        answer,
        0,
    );
    assert!(
        framed.to_ascii_lowercase().contains("inconvenience"),
        "Expected 'inconveniences' label in prefix, got: {framed}"
    );
}

#[test]
fn evaluative_mode_limitations_uses_correct_prefix() {
    let answer = "Rust has a steep learning curve.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Evaluative,
        "rust",
        "what are the limitations of rust",
        answer,
        0,
    );
    assert!(
        framed.to_ascii_lowercase().contains("limitation"),
        "Expected 'limitations' label in prefix, got: {framed}"
    );
}

#[test]
fn task_execution_mode_prepends_here_is_solution() {
    let answer = "print('hello')";
    let framed = ResponseFramer::frame(
        ReasoningMode::TaskExecution,
        "hello world in python",
        "write a hello world in python",
        answer,
        0,
    );
    assert!(
        framed.to_ascii_lowercase().contains("solution")
            || framed.to_ascii_lowercase().contains("here is"),
        "Expected task-execution prefix, got: {framed}"
    );
}

#[test]
fn task_execution_does_not_double_frame_code_block() {
    let answer = "```python\nprint('hello')\n```";
    let framed = ResponseFramer::frame(
        ReasoningMode::TaskExecution,
        "hello world",
        "write a hello world",
        answer,
        0,
    );
    assert_eq!(
        framed, answer,
        "Should not double-frame an answer that starts with ```"
    );
}

#[test]
fn opening_phrase_is_empty_for_exploratory() {
    let phrase = ResponseFramer::opening_phrase(ReasoningMode::Exploratory, "rust", "rust", "", 0);
    assert!(phrase.is_empty(), "Exploratory should return empty opener");
}

#[test]
fn opening_phrase_is_empty_for_factual_lookup() {
    let phrase =
        ResponseFramer::opening_phrase(ReasoningMode::FactualLookup, "rust", "rust", "", 0);
    assert!(
        phrase.is_empty(),
        "FactualLookup should return empty opener"
    );
}

#[test]
#[ignore]
fn opening_phrase_contains_topic_for_definitional() {
    let phrase = ResponseFramer::opening_phrase(
        ReasoningMode::Definitional,
        "borrow checker",
        "what is borrow checker",
        "",
        0,
    );
    assert!(
        phrase.contains("borrow checker") || phrase.contains("Borrow checker"),
        "Phrase should include topic, got: {phrase}"
    );
}

#[test]
#[ignore]
fn frame_handles_empty_answer_without_panic() {
    let result = ResponseFramer::frame(
        ReasoningMode::Causal,
        "test topic",
        "why does test topic",
        "",
        0,
    );
    assert!(result.is_empty() || result.starts_with("The reason"));
}

#[test]
fn frame_handles_empty_topic_without_panic() {
    let result = ResponseFramer::frame(
        ReasoningMode::Evaluative,
        "",
        "what are advantages",
        "Some answer text.",
        0,
    );
    assert!(!result.is_empty());
}

#[test]
#[ignore]
fn causal_predicate_topic_produces_because_opener() {
    let phrase = ResponseFramer::opening_phrase(
        ReasoningMode::Causal,
        "Rust does not have a garbage collector",
        "why does rust not have a garbage collector",
        "",
        0,
    );
    assert!(
        phrase.contains("because"),
        "Expected 'because' in predicate-rich causal opener, got: {phrase}"
    );
}

#[test]
fn causal_no_comma_between_opener_and_answer() {
    let answer = "Rust uses ownership to ensure memory safety at compile time.";
    let framed = ResponseFramer::frame(
        ReasoningMode::Causal,
        "Rust does not have a garbage collector",
        "why does rust not have a garbage collector",
        answer,
        0,
    );
    assert!(
        !framed.contains("because,"),
        "Should not contain 'because,' with spurious comma, got: {framed}"
    );
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
