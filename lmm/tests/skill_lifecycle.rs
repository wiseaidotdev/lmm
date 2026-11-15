// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::skill::{Skill, SkillRegistry};

#[test]
fn skill_upsert_and_get() {
    let mut reg = SkillRegistry::default();
    let skill = Skill {
        topic: "rust".to_string(),
        content: "Rust is a systems programming language.".to_string(),
        confidence: 0.9,
        created_at: 0,
        version: 1,
    };
    reg.upsert(skill);
    let retrieved = reg.get("rust");
    assert!(retrieved.is_some(), "Should retrieve the upserted skill");
    assert_eq!(retrieved.unwrap().topic, "rust");
    assert_eq!(retrieved.unwrap().confidence, 0.9);
}

#[test]
fn skill_confidence_stored() {
    let mut reg = SkillRegistry::default();
    let low = Skill {
        topic: "python".to_string(),
        content: "Python is interpreted.".to_string(),
        confidence: 0.1,
        created_at: 0,
        version: 1,
    };
    let high = Skill {
        topic: "rust".to_string(),
        content: "Rust is fast.".to_string(),
        confidence: 0.9,
        created_at: 0,
        version: 1,
    };
    reg.upsert(low);
    reg.upsert(high);
    assert_eq!(reg.len(), 2);
    assert_eq!(reg.get("python").unwrap().confidence, 0.1);
    assert_eq!(reg.get("rust").unwrap().confidence, 0.9);
}

#[test]
fn skill_topic_matching() {
    let mut reg = SkillRegistry::default();
    let skill = Skill {
        topic: "memory safety in rust".to_string(),
        content: "Rust guarantees memory safety at compile time.".to_string(),
        confidence: 0.85,
        created_at: 0,
        version: 1,
    };
    reg.upsert(skill);
    assert!(
        reg.get("memory safety in rust").is_some(),
        "Exact topic should match"
    );
}

#[test]
fn skill_registry_is_empty_initially() {
    let reg = SkillRegistry::default();
    assert!(reg.is_empty(), "New registry should be empty");
    assert_eq!(reg.len(), 0);
}

#[test]
fn skill_upsert_overwrites_existing() {
    let mut reg = SkillRegistry::default();
    let v1 = Skill {
        topic: "rust".to_string(),
        content: "v1 content".to_string(),
        confidence: 0.5,
        created_at: 0,
        version: 1,
    };
    let v2 = Skill {
        topic: "rust".to_string(),
        content: "v2 content".to_string(),
        confidence: 0.9,
        created_at: 0,
        version: 2,
    };
    reg.upsert(v1);
    reg.upsert(v2);
    assert_eq!(reg.len(), 1, "Upsert should not create duplicates");
    let retrieved = reg.get("rust").unwrap();
    assert_eq!(
        retrieved.content, "v2 content",
        "Should have latest content"
    );
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
