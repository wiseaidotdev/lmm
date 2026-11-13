// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::tram::query::{
    extract_topic, strip_leading_articles, strip_trailing_verb, trim_predicate,
};

#[test]
fn test_extract_topic_simple() {
    assert_eq!(extract_topic("what is rust"), "rust");
    assert_eq!(extract_topic("what is python"), "python");
}

#[test]
fn test_extract_topic_with_article() {
    assert_eq!(
        extract_topic("what is a quantum computer"),
        "quantum computer"
    );
    assert_eq!(extract_topic("what is the blockchain"), "blockchain");
}

#[test]
fn test_extract_topic_evaluative() {
    assert_eq!(extract_topic("advantages of rust"), "rust");
    assert_eq!(extract_topic("benefits of python"), "python");
}

#[test]
fn test_extract_topic_causal() {
    assert_eq!(extract_topic("why does rust not have gc"), "rust");
    assert_eq!(extract_topic("how does tokio work"), "tokio");
}

#[test]
fn test_strip_leading_articles() {
    assert_eq!(strip_leading_articles("a test"), "test");
    assert_eq!(strip_leading_articles("an apple"), "apple");
    assert_eq!(strip_leading_articles("the sun"), "sun");
    assert_eq!(strip_leading_articles("no article"), "no article");
}

#[test]
fn test_trim_predicate() {
    assert_eq!(trim_predicate("rust is fast"), "rust");
    assert_eq!(trim_predicate("python has libraries"), "python");
    assert_eq!(trim_predicate("short"), "short");
}

#[test]
fn test_strip_trailing_verb() {
    assert_eq!(strip_trailing_verb("tokio work".to_string()), "tokio");
    assert_eq!(strip_trailing_verb("python works".to_string()), "python");
    assert_eq!(strip_trailing_verb("no verb".to_string()), "no verb");
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
