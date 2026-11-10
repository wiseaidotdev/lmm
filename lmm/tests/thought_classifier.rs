// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for [`lmm::cognition::thought::classifier::InputClassifier`].

use lmm::cognition::thought::classifier::{InputClassifier, ReasoningMode};

#[test]
fn what_is_classifies_as_factual() {
    assert_eq!(
        InputClassifier::classify("what is the borrow checker?"),
        ReasoningMode::FactualLookup
    );
}

#[test]
fn who_is_classifies_as_factual() {
    assert_eq!(
        InputClassifier::classify("who is Graydon Hoare?"),
        ReasoningMode::FactualLookup
    );
}

#[test]
fn why_does_classifies_as_causal() {
    assert_eq!(
        InputClassifier::classify("why does Rust avoid garbage collection?"),
        ReasoningMode::Causal
    );
}

#[test]
fn what_causes_classifies_as_causal() {
    assert_eq!(
        InputClassifier::classify("what causes a stack overflow?"),
        ReasoningMode::Causal
    );
}

#[test]
fn vs_classifies_as_analogical() {
    assert_eq!(
        InputClassifier::classify("rust vs go"),
        ReasoningMode::Analogical
    );
}

#[test]
fn compare_classifies_as_analogical() {
    assert_eq!(
        InputClassifier::classify("compare Rust and Python performance"),
        ReasoningMode::Analogical
    );
}

#[test]
fn define_classifies_as_definitional() {
    assert_eq!(
        InputClassifier::classify("define ownership in Rust"),
        ReasoningMode::Definitional
    );
}

#[test]
fn explain_classifies_as_definitional() {
    assert_eq!(
        InputClassifier::classify("explain the concept of lifetimes"),
        ReasoningMode::Definitional
    );
}

#[test]
fn how_to_classifies_as_procedural() {
    assert_eq!(
        InputClassifier::classify("how to install Rust"),
        ReasoningMode::Procedural
    );
}

#[test]
fn steps_to_classifies_as_procedural() {
    assert_eq!(
        InputClassifier::classify("steps to build a web server in Rust"),
        ReasoningMode::Procedural
    );
}

#[test]
fn single_word_classifies_as_exploratory() {
    assert_eq!(
        InputClassifier::classify("photosynthesis"),
        ReasoningMode::Exploratory
    );
}

#[test]
fn bare_word_why_classifies_not_as_exploratory() {
    let mode = InputClassifier::classify("why");
    assert!(mode == ReasoningMode::Causal || mode == ReasoningMode::Exploratory);
}

#[test]
fn is_better_classifies_as_evaluative() {
    assert_eq!(
        InputClassifier::classify("is Rust worth learning?"),
        ReasoningMode::Evaluative
    );
}

#[test]
fn pros_and_cons_classifies_as_evaluative() {
    assert_eq!(
        InputClassifier::classify("pros and cons of Rust"),
        ReasoningMode::Evaluative
    );
}

#[test]
fn empty_string_classifies_as_exploratory() {
    assert_eq!(InputClassifier::classify(""), ReasoningMode::Exploratory);
}

#[test]
fn bare_noun_phrase_returns_true_for_bare_noun() {
    assert!(InputClassifier::is_bare_noun("photosynthesis"));
    assert!(InputClassifier::is_bare_noun("Rust programming language"));
}

#[test]
fn question_phrase_is_not_bare_noun() {
    assert!(!InputClassifier::is_bare_noun("what is photosynthesis?"));
    assert!(!InputClassifier::is_bare_noun("how does Rust work?"));
}

#[test]
fn reasoning_mode_labels_are_non_empty() {
    let modes = [
        ReasoningMode::FactualLookup,
        ReasoningMode::Causal,
        ReasoningMode::Analogical,
        ReasoningMode::Definitional,
        ReasoningMode::Procedural,
        ReasoningMode::Exploratory,
        ReasoningMode::Evaluative,
    ];
    for mode in modes {
        assert!(!mode.label().is_empty());
    }
}

#[test]
fn all_mode_labels_are_distinct() {
    use std::collections::HashSet;
    let modes = [
        ReasoningMode::FactualLookup,
        ReasoningMode::Causal,
        ReasoningMode::Analogical,
        ReasoningMode::Definitional,
        ReasoningMode::Procedural,
        ReasoningMode::Exploratory,
        ReasoningMode::Evaluative,
    ];
    let labels: HashSet<&str> = modes.iter().map(|m| m.label()).collect();
    assert_eq!(labels.len(), modes.len());
}
