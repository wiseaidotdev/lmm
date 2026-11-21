// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `LmmAgentBuilder`, fluent builder for [`LmmAgent`].
//!
//! Obtain an instance via [`LmmAgent::builder()`].  Every field has a
//! dedicated setter; call [`build`](LmmAgentBuilder::build) to produce the
//! fully-initialised [`LmmAgent`].

use std::collections::HashSet;

use lmm::cognition::drive::InternalDrive;
use lmm::cognition::knowledge::KnowledgeIndex;
use lmm::cognition::learning::engine::LearningEngine;
use lmm::cognition::types::{
    Capability, ContextManager, Knowledge, Message, Planner, Profile, Reflection, Status, Task,
    TaskScheduler, Tool,
};

use super::LmmAgent;

/// Builder for [`LmmAgent`].
///
/// Obtain via [`LmmAgent::builder()`].
///
/// # Examples
///
/// ```rust
/// use lmm_agent::agent::LmmAgent;
/// use lmm_agent::types::{Message, Planner, Goal};
///
/// let agent = LmmAgent::builder()
///     .persona("Test agent.")
///     .behavior("Tester")
///     .memory(vec![Message::new("user", "Hi")])
///     .planner(Planner {
///         current_plan: vec![Goal {
///             description: "Say hello.".into(),
///             priority: 0,
///             completed: false,
///         }],
///     })
///     .build();
///
/// assert_eq!(agent.persona.as_str(), "Test agent.");
/// assert_eq!(agent.memory.len(), 1);
/// ```
#[derive(Default)]
pub struct LmmAgentBuilder {
    pub(super) id: Option<String>,
    pub(super) persona: Option<String>,
    pub(super) behavior: Option<String>,
    pub(super) status: Option<Status>,
    pub(super) memory: Option<Vec<Message>>,
    pub(super) long_term_memory: Option<Vec<Message>>,
    pub(super) knowledge: Option<Knowledge>,
    pub(super) tools: Option<Vec<Tool>>,
    pub(super) planner: Option<Option<Planner>>,
    pub(super) reflection: Option<Option<Reflection>>,
    pub(super) scheduler: Option<Option<TaskScheduler>>,
    pub(super) profile: Option<Profile>,
    pub(super) context: Option<ContextManager>,
    pub(super) capabilities: Option<HashSet<Capability>>,
    pub(super) tasks: Option<Vec<Task>>,
    pub(super) knowledge_index: Option<KnowledgeIndex>,
    pub(super) learning_engine: Option<Option<LearningEngine>>,
    pub(super) internal_drive: Option<InternalDrive>,
}

impl LmmAgentBuilder {
    /// Sets the agent's unique identifier (default: auto-generated UUIDv4).
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the agent's persona **(required)**.
    pub fn persona(mut self, persona: impl Into<String>) -> Self {
        self.persona = Some(persona.into());
        self
    }

    /// Sets the agent's behavior / role label **(required)**.
    pub fn behavior(mut self, behavior: impl Into<String>) -> Self {
        self.behavior = Some(behavior.into());
        self
    }

    /// Sets the initial [`Status`] (default: [`Status::Idle`]).
    pub fn status(mut self, status: Status) -> Self {
        self.status = Some(status);
        self
    }

    /// Sets the hot memory (default: empty).
    pub fn memory(mut self, memory: Vec<Message>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Sets the long-term memory (default: empty).
    pub fn long_term_memory(mut self, ltm: Vec<Message>) -> Self {
        self.long_term_memory = Some(ltm);
        self
    }

    /// Sets the knowledge base (default: empty).
    pub fn knowledge(mut self, knowledge: Knowledge) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    /// Sets the tool list (default: empty).
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets an optional planner (default: empty planner).
    pub fn planner(mut self, planner: impl Into<Option<Planner>>) -> Self {
        self.planner = Some(planner.into());
        self
    }

    /// Sets an optional reflection module (default: default reflection).
    pub fn reflection(mut self, reflection: impl Into<Option<Reflection>>) -> Self {
        self.reflection = Some(reflection.into());
        self
    }

    /// Sets an optional task scheduler (default: empty scheduler).
    pub fn scheduler(mut self, scheduler: impl Into<Option<TaskScheduler>>) -> Self {
        self.scheduler = Some(scheduler.into());
        self
    }

    /// Sets the profile (default: name = behavior, no traits).
    pub fn profile(mut self, profile: Profile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Sets the context manager (default: empty).
    pub fn context(mut self, context: ContextManager) -> Self {
        self.context = Some(context);
        self
    }

    /// Sets the capability set (default: empty).
    pub fn capabilities(mut self, capabilities: HashSet<Capability>) -> Self {
        self.capabilities = Some(capabilities);
        self
    }

    /// Sets the task queue (default: empty).
    pub fn tasks(mut self, tasks: Vec<Task>) -> Self {
        self.tasks = Some(tasks);
        self
    }

    /// Pre-populates the knowledge index with an existing [`KnowledgeIndex`].
    pub fn knowledge_index(mut self, index: KnowledgeIndex) -> Self {
        self.knowledge_index = Some(index);
        self
    }

    /// Attaches a [`LearningEngine`] for in-environment lifelong learning.
    pub fn learning_engine(mut self, engine: impl Into<Option<LearningEngine>>) -> Self {
        self.learning_engine = Some(engine.into());
        self
    }

    /// Overrides the default [`InternalDrive`] configuration.
    pub fn internal_drive(mut self, drive: InternalDrive) -> Self {
        self.internal_drive = Some(drive);
        self
    }

    /// Constructs the [`LmmAgent`].
    ///
    /// # Panics
    ///
    /// Panics if `persona` or `behavior` were not set.
    pub fn build(self) -> LmmAgent {
        let persona = self
            .persona
            .expect("LmmAgentBuilder: `persona` is required");
        let behavior = self
            .behavior
            .expect("LmmAgentBuilder: `behavior` is required");
        let profile = self.profile.unwrap_or_else(|| Profile {
            name: behavior.clone().into(),
            traits: vec![],
            behavior_script: None,
        });

        LmmAgent {
            id: self.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            persona,
            behavior,
            status: self.status.unwrap_or_default(),
            memory: self.memory.unwrap_or_default(),
            long_term_memory: self.long_term_memory.unwrap_or_default(),
            knowledge: self.knowledge.unwrap_or_default(),
            tools: self.tools.unwrap_or_default(),
            planner: self.planner.unwrap_or_else(|| Some(Planner::default())),
            reflection: self
                .reflection
                .unwrap_or_else(|| Some(Reflection::default())),
            scheduler: self
                .scheduler
                .unwrap_or_else(|| Some(TaskScheduler::default())),
            profile,
            context: self.context.unwrap_or_default(),
            capabilities: self.capabilities.unwrap_or_default(),
            tasks: self.tasks.unwrap_or_default(),
            knowledge_index: self.knowledge_index.unwrap_or_default(),
            learning_engine: self.learning_engine.unwrap_or(None),
            internal_drive: self.internal_drive.unwrap_or_default(),
        }
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
