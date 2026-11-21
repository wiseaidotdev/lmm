// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `LmmAgent` - the core agent struct.
//!
//! `LmmAgent` is the batteries-included foundation for every custom agent.
//! It holds all agent state (hot memory, long-term memory, tools, planner,
//! reflection, scheduler, ...) and provides symbolic text generation powered
//! by `lmm`'s [`TextPredictor`] plus optional DuckDuckGo knowledge enrichment.
//!
//! ## Builder pattern
//!
//! ```rust
//! use lmm_agent::agent::LmmAgent;
//!
//! let agent = LmmAgent::builder()
//!     .persona("Research Assistant")
//!     .behavior("Summarise the Rust ecosystem.")
//!     .build();
//!
//! assert_eq!(agent.persona.as_str(), "Research Assistant");
//! assert_eq!(agent.behavior.as_str(), "Summarise the Rust ecosystem.");
//! ```
//!
//! ## Attribution
//!
//! Adapted from the `autogpt` project's `agents/agent.rs`:
//! <https://github.com/wiseaidotdev/autogpt/blob/main/autogpt/src/agents/agent.rs>

pub mod builder;
mod causal;
mod drive;
mod generate;
mod knowledge;
mod learning;
mod thinking;

pub use builder::LmmAgentBuilder;

use crate::traits::agent::Agent;
use std::borrow::Cow;
use std::collections::HashSet;

use lmm::cognition::drive::InternalDrive;
use lmm::cognition::knowledge::KnowledgeIndex;
use lmm::cognition::learning::engine::LearningEngine;
use lmm::cognition::types::{
    Capability, ContextManager, Knowledge, Message, Planner, Profile, Reflection, Status, Task,
    TaskScheduler, Tool,
};

/// The core agent type.
///
/// Use [`LmmAgent::builder()`] for fluent construction, or
/// [`LmmAgent::new()`] for the quick two-argument form.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::agent::LmmAgent;
///
/// let agent = LmmAgent::builder()
///     .persona("Research Agent")
///     .behavior("Research quantum computing.")
///     .build();
///
/// assert_eq!(agent.persona.as_str(), "Research Agent");
///
/// let agent2 = LmmAgent::new("Scientist".into(), "Do science.".into());
/// assert_eq!(agent2.persona.as_str(), "Scientist");
/// ```
#[derive(Debug, Clone, Default)]
pub struct LmmAgent {
    /// Unique identifier for this agent instance (auto-generated UUIDv4).
    pub id: String,

    /// The primary mission statement for this agent.
    pub persona: String,

    /// The role or behavior label (e.g. `"Research Assistant"`).
    pub behavior: String,

    /// Current lifecycle state.
    pub status: Status,

    /// Hot memory - recent messages kept in RAM.
    pub memory: Vec<Message>,

    /// Long-term memory - persisted between task executions (in-memory store).
    pub long_term_memory: Vec<Message>,

    /// Structured knowledge facts for reasoning.
    pub knowledge: Knowledge,

    /// Callable tools available to this agent.
    pub tools: Vec<Tool>,

    /// Optional goal planner.
    pub planner: Option<Planner>,

    /// Self-reflection / evaluation module.
    pub reflection: Option<Reflection>,

    /// Time-based task scheduler.
    pub scheduler: Option<TaskScheduler>,

    /// Profile traits and behavioural profile.
    pub profile: Profile,

    /// Recent-message context window.
    pub context: ContextManager,

    /// Capabilities the agent possesses.
    pub capabilities: HashSet<Capability>,

    /// Active task queue.
    pub tasks: Vec<Task>,

    /// Queryable knowledge base built from ingested documents or URLs.
    pub knowledge_index: KnowledgeIndex,

    /// Optional HELM learning engine for in-environment lifelong learning.
    pub learning_engine: Option<LearningEngine>,

    /// Internalized drive system for intrinsic motivation signals.
    pub internal_drive: InternalDrive,
}

/// Core constructors and simple memory/planning methods.
impl LmmAgent {
    /// Returns a new [`LmmAgentBuilder`].
    ///
    /// The builder accepts every field with setter methods and calls
    /// `.build()` to produce the final [`LmmAgent`].
    pub fn builder() -> LmmAgentBuilder {
        LmmAgentBuilder::default()
    }

    /// Constructs an [`LmmAgent`] with the given persona and behavior;
    /// every other field is set to its sensible default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    ///
    /// let agent = LmmAgent::new("Researcher".into(), "Research Rust.".into());
    /// assert_eq!(agent.behavior.as_str(), "Research Rust.");
    /// ```
    pub fn new(persona: Cow<'static, str>, behavior: Cow<'static, str>) -> Self {
        LmmAgent::builder()
            .persona(persona.into_owned())
            .behavior(behavior.into_owned())
            .build()
    }

    /// Appends a [`Message`] to the agent's hot memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    /// use lmm_agent::types::Message;
    ///
    /// let mut agent = LmmAgent::new("Tester".into(), "Test.".into());
    /// agent.add_message(Message::new("user", "Hello"));
    /// assert_eq!(agent.memory.len(), 1);
    /// ```
    pub fn add_message(&mut self, message: Message) {
        self.memory.push(message);
    }

    /// Appends a [`Message`] to the agent's long-term memory.
    pub fn add_ltm_message(&mut self, message: Message) {
        self.long_term_memory.push(message);
    }

    /// Marks a goal as completed by its description substring.
    ///
    /// Returns `true` if a matching goal was found and updated.
    pub fn complete_goal(&mut self, description_substr: &str) -> bool {
        if let Some(plan) = self.planner.as_mut() {
            for goal in &mut plan.current_plan {
                if goal.description.contains(description_substr) {
                    goal.completed = true;
                    return true;
                }
            }
        }
        false
    }
}

/// [`Agent`] trait implementation, delegates to the inherent methods above.
impl Agent for LmmAgent {
    fn new(persona: Cow<'static, str>, behavior: Cow<'static, str>) -> Self {
        LmmAgent::new(persona, behavior)
    }

    fn update(&mut self, status: Status) {
        self.status = status;
    }

    fn persona(&self) -> &str {
        &self.persona
    }

    fn behavior(&self) -> &str {
        &self.behavior
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn memory(&self) -> &Vec<Message> {
        &self.memory
    }

    fn tools(&self) -> &Vec<Tool> {
        &self.tools
    }

    fn knowledge(&self) -> &Knowledge {
        &self.knowledge
    }

    fn planner(&self) -> Option<&Planner> {
        self.planner.as_ref()
    }

    fn profile(&self) -> &Profile {
        &self.profile
    }

    fn reflection(&self) -> Option<&Reflection> {
        self.reflection.as_ref()
    }

    fn scheduler(&self) -> Option<&TaskScheduler> {
        self.scheduler.as_ref()
    }

    fn capabilities(&self) -> &HashSet<Capability> {
        &self.capabilities
    }

    fn context(&self) -> &ContextManager {
        &self.context
    }

    fn tasks(&self) -> &Vec<Task> {
        &self.tasks
    }

    fn memory_mut(&mut self) -> &mut Vec<Message> {
        &mut self.memory
    }

    fn planner_mut(&mut self) -> Option<&mut Planner> {
        self.planner.as_mut()
    }

    fn context_mut(&mut self) -> &mut ContextManager {
        &mut self.context
    }
}

impl lmm::cognition::types::AgentEval for LmmAgent {
    fn planner(&self) -> Option<&Planner> {
        self.planner.as_ref()
    }

    fn memory(&self) -> &[Message] {
        &self.memory
    }

    fn knowledge(&self) -> Option<&Knowledge> {
        Some(&self.knowledge)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
