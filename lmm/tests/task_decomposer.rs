// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use lmm::cognition::task_decomposer::{TaskDecomposer, TaskKind};
use lmm::cognition::thought::classifier::ReasoningMode;
use lmm::constants::task_decomposer::{TASK_LOW_REWARD_THRESHOLD, TASK_MEDIUM_REWARD_THRESHOLD};

#[test]
fn decompose_intelligent_medium_reward_adds_refinement_queries() {
    let mid_reward = (TASK_LOW_REWARD_THRESHOLD + TASK_MEDIUM_REWARD_THRESHOLD) / 2.0;
    let tasks = TaskDecomposer::decompose_intelligent(
        "what are the advantages of rust?",
        ReasoningMode::Evaluative,
        9,
        mid_reward,
    );

    assert!(
        !tasks.is_empty(),
        "Should produce tasks for medium reward={mid_reward}"
    );

    assert!(
        !matches!(tasks[0].kind, TaskKind::AcquireSkill { .. }),
        "First task must NOT be AcquireSkill at medium reward={mid_reward}"
    );

    let has_refinement = tasks.iter().any(|t| {
        let q = t.query.to_ascii_lowercase();
        q.contains("what is") || q.contains("definition") || q.contains("examples")
    });
    assert!(
        has_refinement,
        "Medium-reward decomposition should include definitional/refinement queries; \
         got: {:?}",
        tasks.iter().map(|t| &t.query).collect::<Vec<_>>()
    );
}

#[test]
fn decompose_intelligent_medium_reward_does_not_exceed_max() {
    let mid_reward = (TASK_LOW_REWARD_THRESHOLD + TASK_MEDIUM_REWARD_THRESHOLD) / 2.0;
    let max = 4usize;
    let tasks = TaskDecomposer::decompose_intelligent(
        "what are the advantages of rust?",
        ReasoningMode::Evaluative,
        max,
        mid_reward,
    );
    assert!(
        tasks.len() <= max,
        "Medium-reward decomposition must respect max={max}, got {}",
        tasks.len()
    );
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
