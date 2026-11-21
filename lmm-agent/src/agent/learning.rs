// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Learning engine integration methods for [`LmmAgent`].
//!
//! Covers [`save_learning`](LmmAgent::save_learning),
//! [`load_learning`](LmmAgent::load_learning), and
//! [`recall_learned`](LmmAgent::recall_learned).

use anyhow::Result;
use lmm::cognition::learning::q_table::{ActionKey, QTable};
use lmm::cognition::learning::store::LearningStore;

use super::LmmAgent;

impl LmmAgent {
    /// Saves the current [`LearningEngine`] state to `path` as JSON.
    ///
    /// Returns `Ok(())` when no learning engine is attached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    /// use lmm::cognition::learning::engine::LearningEngine;
    /// use lmm::cognition::learning::config::LearningConfig;
    ///
    /// let mut agent = LmmAgent::builder()
    ///     .persona("Learner")
    ///     .behavior("Learn.")
    ///     .learning_engine(LearningEngine::new(LearningConfig::default()))
    ///     .build();
    ///
    /// let path = std::env::temp_dir().join(format!("agent_helm_{}.json", uuid::Uuid::new_v4()));
    /// agent.save_learning(&path).unwrap();
    /// ```
    pub fn save_learning(&self, path: &std::path::Path) -> Result<()> {
        if let Some(engine) = &self.learning_engine {
            LearningStore::save(engine, path)
        } else {
            Ok(())
        }
    }

    /// Loads a previously saved [`LearningEngine`] state from `path` and
    /// attaches it to this agent, replacing any existing engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    /// use lmm::cognition::learning::engine::LearningEngine;
    /// use lmm::cognition::learning::config::LearningConfig;
    ///
    /// let mut agent = LmmAgent::builder()
    ///     .persona("Learner")
    ///     .behavior("Learn.")
    ///     .learning_engine(LearningEngine::new(LearningConfig::default()))
    ///     .build();
    ///
    /// let path = std::env::temp_dir().join(format!("agent_helm_load_{}.json", uuid::Uuid::new_v4()));
    /// agent.save_learning(&path).unwrap();
    /// agent.load_learning(&path).unwrap();
    /// ```
    pub fn load_learning(&mut self, path: &std::path::Path) -> Result<()> {
        let engine = LearningStore::load(path)?;
        self.learning_engine = Some(engine);
        Ok(())
    }

    /// Returns the Q-table-recommended action for the current query string,
    /// or `None` when no learning engine is attached or the state is unknown.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    /// use lmm::cognition::learning::engine::LearningEngine;
    /// use lmm::cognition::learning::config::LearningConfig;
    ///
    /// let mut agent = LmmAgent::builder()
    ///     .persona("Learner")
    ///     .behavior("Learn.")
    ///     .learning_engine(LearningEngine::new(LearningConfig::default()))
    ///     .build();
    ///
    /// let action = agent.recall_learned("rust memory safety", 0);
    /// // No experience recorded yet, so the engine explores freely.
    /// assert!(action.is_some());
    /// ```
    pub fn recall_learned(&mut self, query: &str, step: usize) -> Option<ActionKey> {
        let engine = self.learning_engine.as_mut()?;
        let state = QTable::state_key(query);
        Some(engine.recommend_action(state, query, step))
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
