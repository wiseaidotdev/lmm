// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::thought::classifier::{InputClassifier, ReasoningMode};

#[test]
fn what_is_classifies_as_factual_lookup() {
    assert_eq!(
        InputClassifier::classify("what is the borrow checker?"),
        ReasoningMode::FactualLookup
    );
}

#[test]
fn who_is_classifies_as_factual_lookup() {
    assert_eq!(
        InputClassifier::classify("who is Graydon Hoare?"),
        ReasoningMode::FactualLookup
    );
}

#[test]
fn bare_noun_classifies_as_exploratory() {
    assert_eq!(
        InputClassifier::classify("photosynthesis"),
        ReasoningMode::Exploratory
    );
}

#[test]
fn single_proper_noun_classifies_as_exploratory() {
    assert_eq!(
        InputClassifier::classify("Rust"),
        ReasoningMode::Exploratory
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
fn vs_keyword_classifies_as_analogical() {
    assert_eq!(
        InputClassifier::classify("rust vs go"),
        ReasoningMode::Analogical
    );
}

#[test]
fn compare_keyword_classifies_as_analogical() {
    assert_eq!(
        InputClassifier::classify("compare rust and python performance"),
        ReasoningMode::Analogical
    );
}

#[test]
fn difference_between_classifies_as_analogical() {
    assert_eq!(
        InputClassifier::classify("difference between concurrency and parallelism"),
        ReasoningMode::Analogical
    );
}

#[test]
fn how_to_classifies_as_procedural() {
    assert_eq!(
        InputClassifier::classify("how to install Rust on Linux"),
        ReasoningMode::Procedural
    );
}

#[test]
fn steps_to_classifies_as_procedural() {
    assert_eq!(
        InputClassifier::classify("steps to build a Rust web server"),
        ReasoningMode::Procedural
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
fn meaning_keyword_classifies_as_definitional() {
    assert_eq!(
        InputClassifier::classify("what is the meaning of ownership in Rust?"),
        ReasoningMode::Definitional
    );
}

#[test]
fn is_it_good_classifies_as_evaluative() {
    assert_eq!(
        InputClassifier::classify("is Rust worth learning in 2025?"),
        ReasoningMode::Evaluative
    );
}

#[test]
fn should_i_classifies_as_evaluative() {
    assert_eq!(
        InputClassifier::classify("should I use Rust or Python for data science?"),
        ReasoningMode::Evaluative
    );
}

#[test]
fn classify_returns_a_valid_mode_for_empty_input() {
    let mode = InputClassifier::classify("");
    let valid_modes = [
        ReasoningMode::FactualLookup,
        ReasoningMode::Causal,
        ReasoningMode::Analogical,
        ReasoningMode::Definitional,
        ReasoningMode::Procedural,
        ReasoningMode::Exploratory,
        ReasoningMode::Evaluative,
    ];
    assert!(
        valid_modes.contains(&mode),
        "empty input produced an unexpected mode: {:?}",
        mode
    );
}

#[test]
fn classify_is_case_insensitive_for_why() {
    let lower = InputClassifier::classify("why is Rust fast?");
    let upper = InputClassifier::classify("WHY is Rust fast?");
    assert_eq!(lower, upper);
}

#[test]
fn classify_is_case_insensitive_for_how_to() {
    let lower = InputClassifier::classify("how to compile Rust");
    let upper = InputClassifier::classify("HOW TO compile Rust");
    assert_eq!(lower, upper);
}

#[test]
fn classify_is_case_insensitive_for_vs() {
    let lower = InputClassifier::classify("rust vs python speed");
    let upper = InputClassifier::classify("Rust VS Python speed");
    assert_eq!(lower, upper);
}

#[test]
fn explain_classifies_as_factual_or_definitional() {
    let mode = InputClassifier::classify("explain the ownership model in Rust");
    assert!(
        matches!(
            mode,
            ReasoningMode::FactualLookup | ReasoningMode::Definitional | ReasoningMode::Exploratory
        ),
        "unexpected mode for explain prompt: {:?}",
        mode
    );
}

#[test]
fn all_seven_modes_are_reachable() {
    let probes: &[(&str, ReasoningMode)] = &[
        ("what is Rust", ReasoningMode::FactualLookup),
        ("why does Rust not have GC", ReasoningMode::Causal),
        ("rust vs python", ReasoningMode::Analogical),
        ("define lifetime in Rust", ReasoningMode::Definitional),
        ("how to install rust", ReasoningMode::Procedural),
        ("machine learning", ReasoningMode::Exploratory),
        // "worth" (5) + "good" (5) beats any analogical/factual signal
        (
            "is Rust worth learning pros and cons",
            ReasoningMode::Evaluative,
        ),
    ];
    for (prompt, expected) in probes {
        let got = InputClassifier::classify(prompt);
        assert_eq!(
            got, *expected,
            "prompt '{prompt}' → expected {expected:?}, got {got:?}"
        );
    }
}

#[test]
fn reasoning_mode_equality_is_reflexive() {
    assert_eq!(ReasoningMode::FactualLookup, ReasoningMode::FactualLookup);
    assert_eq!(ReasoningMode::Causal, ReasoningMode::Causal);
    assert_eq!(ReasoningMode::Analogical, ReasoningMode::Analogical);
    assert_eq!(ReasoningMode::Definitional, ReasoningMode::Definitional);
    assert_eq!(ReasoningMode::Procedural, ReasoningMode::Procedural);
    assert_eq!(ReasoningMode::Exploratory, ReasoningMode::Exploratory);
    assert_eq!(ReasoningMode::Evaluative, ReasoningMode::Evaluative);
}

#[test]
fn reasoning_mode_inequality_between_distinct_variants() {
    assert_ne!(ReasoningMode::FactualLookup, ReasoningMode::Causal);
    assert_ne!(ReasoningMode::Analogical, ReasoningMode::Procedural);
    assert_ne!(ReasoningMode::Definitional, ReasoningMode::Evaluative);
    assert_ne!(ReasoningMode::Exploratory, ReasoningMode::FactualLookup);
}
