// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::generation::filter::SentenceFilter;

#[test]
fn accepts_quality_factual_prose() {
    let s = "Rust is a general-purpose programming language that emphasizes performance, type safety, and memory safety.";
    assert!(SentenceFilter::is_acceptable(s));
}

#[test]
fn accepts_sentence_with_named_subject_and_date() {
    let s =
        "Software developer Graydon Hoare created Rust in 2006 while working at Mozilla Research.";
    assert!(SentenceFilter::is_acceptable(s));
}

#[test]
fn accepts_long_descriptive_sentence_about_python() {
    let s = "Python is a high-level, general-purpose programming language that emphasizes code readability, simplicity, and ease-of-writing with significant indentation.";
    assert!(SentenceFilter::is_acceptable(s));
}

#[test]
fn rejects_category_lines() {
    let s = "Category History of programming languages - The history of programming languages spans from early computers.";
    assert!(!SentenceFilter::is_acceptable(s));
}

#[test]
fn rejects_category_colon_prefix() {
    let s = "Category:Rust programming language articles lacking sources.";
    assert!(!SentenceFilter::is_acceptable(s));
}

#[test]
fn rejects_outline_fragments() {
    let s = "Outline of the Rust programming language - The following outline is provided as an overview of Rust.";
    assert!(!SentenceFilter::is_acceptable(s));
}

#[test]
fn rejects_disambiguation_see_also() {
    let s = "See also Rust disambiguation for the many things this word can refer to.";
    assert!(!SentenceFilter::is_acceptable(s));
}

#[test]
fn rejects_too_short_sentences() {
    let s = "Rust is fast.";
    assert!(!SentenceFilter::is_acceptable(s));
}

#[test]
fn rejects_empty_string() {
    assert!(!SentenceFilter::is_acceptable(""));
}

#[test]
fn rejects_whitespace_only() {
    assert!(!SentenceFilter::is_acceptable("   "));
}

#[test]
fn rejects_low_density_repetition() {
    let s = "rust rust rust rust rust rust rust rust rust rust rust rust rust.";
    assert!(!SentenceFilter::is_acceptable(s));
}

#[test]
fn rejects_header_fragment_with_dash() {
    let s = "Rust programming language - The history of its development started in 2006 at Mozilla Research.";
    assert!(!SentenceFilter::is_acceptable(s));
}

#[test]
fn is_bare_question_title_detects_what_without_question_mark() {
    let s = "What is the difference between concurrency and parallelism";
    assert!(SentenceFilter::is_bare_question_title(s));
}

#[test]
fn is_bare_question_title_detects_how_without_question_mark() {
    let s = "How does the borrow checker enforce memory safety";
    assert!(SentenceFilter::is_bare_question_title(s));
}

#[test]
fn is_bare_question_title_detects_why_without_question_mark() {
    let s = "Why does Rust not have a garbage collector";
    assert!(SentenceFilter::is_bare_question_title(s));
}

#[test]
fn is_bare_question_title_detects_when_without_question_mark() {
    let s = "When was Python first released to the public by Guido van Rossum";
    assert!(SentenceFilter::is_bare_question_title(s));
}

#[test]
fn is_bare_question_title_allows_proper_question_ending_with_mark() {
    let s = "What is Rust?";
    assert!(!SentenceFilter::is_bare_question_title(s));
}

#[test]
fn is_acceptable_rejects_bare_question_title() {
    let s = "What does the ownership model in Rust prevent at compile time without a runtime";
    assert!(!SentenceFilter::is_acceptable(s));
}

#[test]
fn is_acceptable_allows_question_ending_with_mark() {
    let _ = SentenceFilter::is_acceptable("What is Rust?");
}

#[test]
fn has_minimum_length_accepts_eight_words() {
    let s = "Rust is a systems programming language for memory safety.";
    assert!(SentenceFilter::has_minimum_length(s));
}

#[test]
fn has_minimum_length_rejects_seven_words() {
    let s = "Rust is a programming language for safety.";
    assert!(!SentenceFilter::has_minimum_length(s));
}

#[test]
fn is_low_density_flags_repetitive_text() {
    let s = "rust rust rust rust rust rust rust rust rust rust rust.";
    assert!(SentenceFilter::is_low_density(s));
}

#[test]
fn is_low_density_does_not_flag_varied_prose() {
    let s = "Rust is a systems programming language that prioritizes safety, speed, and concurrency without a garbage collector.";
    assert!(!SentenceFilter::is_low_density(s));
}

#[test]
fn is_header_fragment_detects_dash_pattern() {
    let s = "Python programming language - A widely-used high-level programming language.";
    assert!(SentenceFilter::is_header_fragment(s));
}

#[test]
fn is_header_fragment_does_not_flag_normal_prose() {
    let s = "Rust provides memory safety guarantees without requiring a garbage collector.";
    assert!(!SentenceFilter::is_header_fragment(s));
}

#[test]
fn is_metadata_line_detects_category() {
    assert!(SentenceFilter::is_metadata_line(
        "Category some article here"
    ));
}

#[test]
fn is_metadata_line_detects_outline() {
    assert!(SentenceFilter::is_metadata_line(
        "Outline of something - The following outline is a topical guide."
    ));
}

#[test]
fn is_disambiguation_detects_see_also() {
    assert!(SentenceFilter::is_disambiguation_fragment(
        "See also the main article for disambiguation purposes."
    ));
}

#[test]
fn is_good_opener_accepts_named_subject() {
    assert!(SentenceFilter::is_good_opener(
        "Rust is a general-purpose programming language emphasizing performance and memory safety."
    ));
}

#[test]
fn is_good_opener_rejects_pronoun_it() {
    assert!(!SentenceFilter::is_good_opener(
        "It was influenced by ideas from functional programming including immutability."
    ));
}

#[test]
fn is_good_opener_rejects_pronoun_they() {
    assert!(!SentenceFilter::is_good_opener(
        "They designed the language to prevent data races at compile time by default."
    ));
}

#[test]
fn is_good_opener_rejects_continuation_adverb_also() {
    assert!(!SentenceFilter::is_good_opener(
        "Rust also provides traits and methods as alternatives to object-oriented inheritance."
    ));
}

#[test]
fn is_good_opener_rejects_by_as_first_word() {
    assert!(!SentenceFilter::is_good_opener(
        "By extension the borrow checker enforces memory safety without a garbage collector."
    ));
}
