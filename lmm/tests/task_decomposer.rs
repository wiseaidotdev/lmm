// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::task_decomposer::TaskDecomposer;
use lmm::cognition::thought::classifier::ReasoningMode;

#[test]
fn decompose_evaluative_produces_advantage_queries() {
    let tasks = TaskDecomposer::decompose(
        "what are the advantages of rust?",
        ReasoningMode::Evaluative,
        4,
    );
    assert!(!tasks.is_empty(), "Should produce at least one task");
    let queries: Vec<&str> = tasks.iter().map(|t| t.query.as_str()).collect();
    assert!(
        queries
            .iter()
            .any(|q| q.contains("advantage") || q.contains("benefit")),
        "Should have advantage/benefit query"
    );
}

#[test]
fn decompose_evaluative_includes_disadvantage_query() {
    let tasks = TaskDecomposer::decompose("pros and cons of rust", ReasoningMode::Evaluative, 4);
    let queries: Vec<&str> = tasks.iter().map(|t| t.query.as_str()).collect();
    assert!(
        queries
            .iter()
            .any(|q| q.contains("disadvantage") || q.contains("pros and cons")),
        "Should have disadvantage/pros-and-cons query, got: {:?}",
        queries
    );
}

#[test]
fn decompose_task_execution_contains_code_generation_task() {
    use lmm::cognition::task_decomposer::TaskKind;
    let tasks = TaskDecomposer::decompose(
        "write hello world in python",
        ReasoningMode::TaskExecution,
        4,
    );
    assert!(!tasks.is_empty());
    let has_code_gen = tasks
        .iter()
        .any(|t| matches!(&t.kind, TaskKind::CodeGeneration { .. }));
    assert!(
        has_code_gen,
        "TaskExecution should include CodeGeneration task"
    );
}

#[test]
fn decompose_task_execution_extracts_python_language() {
    use lmm::cognition::task_decomposer::TaskKind;
    let tasks = TaskDecomposer::decompose(
        "write hello world in python",
        ReasoningMode::TaskExecution,
        4,
    );
    let lang = tasks.iter().find_map(|t| {
        if let TaskKind::CodeGeneration { language, .. } = &t.kind {
            Some(language.clone())
        } else {
            None
        }
    });
    assert_eq!(lang.as_deref(), Some("python"), "Should extract 'python'");
}

#[test]
fn decompose_causal_produces_why_query() {
    let tasks = TaskDecomposer::decompose(
        "why does rust avoid garbage collection",
        ReasoningMode::Causal,
        4,
    );
    assert!(!tasks.is_empty());
    let has_why = tasks.iter().any(|t| t.query.contains("why"));
    assert!(has_why, "Causal decompose should include a 'why' query");
}

#[test]
fn decompose_analogical_produces_vs_query() {
    let tasks = TaskDecomposer::decompose("rust vs python", ReasoningMode::Analogical, 4);
    assert!(!tasks.is_empty());
    let has_vs = tasks
        .iter()
        .any(|t| t.query.contains("rust") && t.query.contains("python"));
    assert!(has_vs, "Analogical should produce comparison query");
}

#[test]
fn decompose_analogical_produces_comparison_task_kind() {
    use lmm::cognition::task_decomposer::TaskKind;
    let tasks = TaskDecomposer::decompose("rust vs python", ReasoningMode::Analogical, 4);
    let has_comparison = tasks
        .iter()
        .any(|t| matches!(&t.kind, TaskKind::Comparison(..)));
    assert!(
        has_comparison,
        "Should have Comparison task kind for vs prompt"
    );
}

#[test]
fn decompose_procedural_produces_how_to_query() {
    let tasks = TaskDecomposer::decompose("how to install rust", ReasoningMode::Procedural, 4);
    assert!(!tasks.is_empty());
    let has_how = tasks
        .iter()
        .any(|t| t.query.contains("how to") || t.query.contains("tutorial"));
    assert!(
        has_how,
        "Procedural should produce how-to or tutorial query"
    );
}

#[test]
fn decompose_definitional_produces_definition_task_kind() {
    use lmm::cognition::task_decomposer::TaskKind;
    let tasks =
        TaskDecomposer::decompose("define ownership in rust", ReasoningMode::Definitional, 4);
    assert!(!tasks.is_empty());
    let has_def = tasks
        .iter()
        .any(|t| matches!(&t.kind, TaskKind::Definition(..)));
    assert!(has_def, "Definitional should include Definition task kind");
}

#[test]
fn decompose_respects_max_limit() {
    let tasks = TaskDecomposer::decompose(
        "what are the advantages of rust?",
        ReasoningMode::Evaluative,
        2,
    );
    assert!(tasks.len() <= 2, "Should not exceed max_tasks=2");
}

#[test]
fn decompose_returns_at_least_one_task_for_all_modes() {
    for mode in [
        ReasoningMode::FactualLookup,
        ReasoningMode::Causal,
        ReasoningMode::Analogical,
        ReasoningMode::Definitional,
        ReasoningMode::Procedural,
        ReasoningMode::Exploratory,
        ReasoningMode::Evaluative,
        ReasoningMode::TaskExecution,
    ] {
        let tasks = TaskDecomposer::decompose("rust programming", mode, 4);
        assert!(!tasks.is_empty(), "Mode {:?} should produce tasks", mode);
    }
}

#[test]
fn is_creative_task_detects_write_prefix() {
    assert!(TaskDecomposer::is_creative_task(
        "write hello world in python"
    ));
}

#[test]
fn is_creative_task_detects_generate_prefix() {
    assert!(TaskDecomposer::is_creative_task(
        "generate a fibonacci function in rust"
    ));
}

#[test]
fn is_creative_task_detects_create_prefix() {
    assert!(TaskDecomposer::is_creative_task(
        "create a sorting algorithm"
    ));
}

#[test]
fn is_creative_task_returns_false_for_question() {
    assert!(!TaskDecomposer::is_creative_task("what is rust?"));
}

#[test]
fn is_creative_task_returns_false_for_bare_noun() {
    assert!(!TaskDecomposer::is_creative_task("rust programming"));
}

#[test]
fn extract_code_language_finds_python() {
    assert_eq!(
        TaskDecomposer::extract_code_language("write hello world in python"),
        Some("python".to_string())
    );
}

#[test]
fn extract_code_language_finds_rust() {
    assert_eq!(
        TaskDecomposer::extract_code_language("implement fibonacci in rust"),
        Some("rust".to_string())
    );
}

#[test]
fn extract_code_language_finds_javascript() {
    assert_eq!(
        TaskDecomposer::extract_code_language("write a function in javascript"),
        Some("javascript".to_string())
    );
}

#[test]
fn extract_code_language_finds_go() {
    assert_eq!(
        TaskDecomposer::extract_code_language("hello world in golang"),
        Some("go".to_string())
    );
}

#[test]
fn extract_code_language_returns_none_for_no_language() {
    assert_eq!(
        TaskDecomposer::extract_code_language("write a sorting algorithm"),
        None
    );
}

#[test]
fn task_priorities_are_ordered_ascending() {
    let tasks = TaskDecomposer::decompose(
        "what are the advantages of rust?",
        ReasoningMode::Evaluative,
        4,
    );
    let priorities: Vec<u8> = tasks.iter().map(|t| t.priority).collect();
    let mut sorted = priorities.clone();
    sorted.sort();
    assert_eq!(
        priorities, sorted,
        "Tasks should be ordered by ascending priority"
    );
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
