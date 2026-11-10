// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::generation::filter::SentenceFilter;

#[test]
fn filter_accepts_quality_prose() {
    let sentence = "Rust is a general-purpose programming language that emphasizes performance, type safety, and memory safety.";
    assert!(SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_accepts_factual_sentence_with_attribution() {
    let sentence =
        "Software developer Graydon Hoare created Rust in 2006 while working at Mozilla.";
    assert!(SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_rejects_category_lines() {
    let sentence = "Category History of programming languages - The history of programming languages spans from early computers.";
    assert!(!SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_rejects_category_prefix() {
    let sentence = "Category:Rust programming language articles lacking sources.";
    assert!(!SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_rejects_outline_fragments() {
    let sentence = "Outline of the Rust programming language - The following outline is provided as an overview of Rust.";
    assert!(!SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_rejects_disambiguation_fragments() {
    let sentence = "Rust An iron oxide, a usually reddish-brown oxide formed by the reaction of iron and oxygen.";
    assert!(!SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_rejects_disambiguation_see_also() {
    let sentence = "See also Rust disambiguation for the many things this word can refer to.";
    assert!(!SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_rejects_short_sentences() {
    let sentence = "Rust is fast.";
    assert!(!SentenceFilter::has_minimum_length(sentence));
    assert!(!SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_rejects_empty_string() {
    assert!(!SentenceFilter::is_acceptable(""));
}

#[test]
fn filter_rejects_whitespace_only() {
    assert!(!SentenceFilter::is_acceptable("   "));
}

#[test]
fn filter_rejects_low_density_repetition() {
    let sentence = "rust rust rust rust rust rust rust rust rust rust rust rust rust.";
    assert!(SentenceFilter::is_low_density(sentence));
    assert!(!SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_rejects_header_fragment_with_dash() {
    let sentence = "Rust programming language - The history of its development started in 2006 at Mozilla Research.";
    assert!(SentenceFilter::is_header_fragment(sentence));
    assert!(!SentenceFilter::is_acceptable(sentence));
}

#[test]
fn filter_is_metadata_line_detects_category() {
    assert!(SentenceFilter::is_metadata_line(
        "Category some article here"
    ));
}

#[test]
fn filter_is_metadata_line_detects_outline() {
    assert!(SentenceFilter::is_metadata_line(
        "Outline of something - The following outline is a topical guide."
    ));
}

#[test]
fn filter_is_disambiguation_detects_see_also() {
    assert!(SentenceFilter::is_disambiguation_fragment(
        "See also the main article for disambiguation purposes."
    ));
}

#[test]
fn filter_has_minimum_length_accepts_eight_words() {
    let sentence = "Rust is a systems programming language for memory safety.";
    assert!(SentenceFilter::has_minimum_length(sentence));
}

#[test]
fn filter_has_minimum_length_rejects_seven_words() {
    let sentence = "Rust is a programming language for safety.";
    assert!(!SentenceFilter::has_minimum_length(sentence));
}

#[test]
fn opener_accepts_named_subject_start() {
    assert!(SentenceFilter::is_good_opener(
        "Rust is a general-purpose programming language emphasizing performance and memory safety."
    ));
}

#[test]
fn opener_accepts_description_start() {
    assert!(SentenceFilter::is_good_opener(
        "Software developer Graydon Hoare created Rust in 2006 while working at Mozilla Research."
    ));
}

#[test]
fn opener_rejects_pronoun_it() {
    assert!(!SentenceFilter::is_good_opener(
        "It was influenced by ideas from functional programming including immutability."
    ));
}

#[test]
fn opener_rejects_pronoun_they() {
    assert!(!SentenceFilter::is_good_opener(
        "They designed the language to prevent data races at compile time by default."
    ));
}

#[test]
fn opener_rejects_pronoun_this() {
    assert!(!SentenceFilter::is_good_opener(
        "This makes it possible to write safe concurrent code without a runtime."
    ));
}

#[test]
fn opener_rejects_continuation_adverb_also_as_second_word() {
    assert!(!SentenceFilter::is_good_opener(
        "Rust also provides traits and methods as alternatives to object-oriented inheritance."
    ));
}

#[test]
fn opener_rejects_continuation_adverb_still_as_second_word() {
    assert!(!SentenceFilter::is_good_opener(
        "Python still supports dynamic typing and is widely used for data science tasks."
    ));
}

#[test]
fn opener_rejects_by_as_first_word() {
    assert!(!SentenceFilter::is_good_opener(
        "By extension the borrow checker enforces memory safety without a garbage collector."
    ));
}

#[test]
fn opener_rejects_both_as_first_word() {
    assert!(!SentenceFilter::is_good_opener(
        "Both ownership and borrowing rules are enforced at compile time by the borrow checker."
    ));
}

#[test]
fn opener_accepts_named_proper_noun_no_continuation() {
    assert!(SentenceFilter::is_good_opener(
        "The Rust Foundation sponsors the ongoing development of the language and its ecosystem."
    ));
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
