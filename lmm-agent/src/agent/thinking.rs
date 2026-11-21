// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # ThinkLoop reasoning methods for [`LmmAgent`].
//!
//! Exposes [`think`](LmmAgent::think) and [`think_with`](LmmAgent::think_with).

use anyhow::Result;
use lmm::cognition::learning::q_table::QTable;
use lmm::cognition::r#loop::ThinkLoop;
use lmm::cognition::search::SearchOracle;
use lmm::cognition::types::{Message, Status, ThinkResult};

use super::LmmAgent;

impl LmmAgent {
    /// Runs the closed-loop **ThinkLoop** reasoning cycle toward `goal`.
    ///
    /// The agent transitions through `Status::Thinking` and back to
    /// `Status::Completed`. At the end of the run the cold-store archive is
    /// serialised into the agent's `long_term_memory` so knowledge persists
    /// across multiple `think()` calls.
    ///
    /// ## Parameters
    ///
    /// * `goal` - natural-language task description (the setpoint).
    ///
    /// Defaults used internally:
    /// - `max_iterations = 10`
    /// - `convergence_threshold = 0.25`
    /// - `k_proportional = 1.0`
    /// - `k_integral = 0.05`
    ///
    /// Use [`LmmAgent::think_with`] for fine-grained control.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[tokio::main]
    /// async fn main() {
    ///     use lmm_agent::agent::LmmAgent;
    ///
    ///     let mut agent = LmmAgent::new("Researcher".into(), "Explore Rust.".into());
    ///     let result = agent.think("What is Rust ownership?").await.unwrap();
    ///     assert!(result.steps > 0);
    ///     assert!(result.final_error >= 0.0 && result.final_error <= 1.0);
    /// }
    /// ```
    pub async fn think(&mut self, goal: &str) -> Result<ThinkResult> {
        self.think_with(goal, 10, 0.25, 1.0, 0.05).await
    }

    /// Like [`think`](Self::think) but exposes all ThinkLoop parameters.
    ///
    /// # Arguments
    ///
    /// * `goal`                  - natural-language goal / setpoint.
    /// * `max_iterations`        - maximum feedback loop iterations (≥ 1).
    /// * `convergence_threshold` - Jaccard error threshold ∈ [0, 1].
    /// * `k_proportional`        - proportional gain Kp.
    /// * `k_integral`            - integral gain Ki.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[tokio::main]
    /// async fn main() {
    ///     use lmm_agent::agent::LmmAgent;
    ///
    ///     let mut agent = LmmAgent::new("Researcher".into(), "Explore Rust.".into());
    ///     let result = agent
    ///         .think_with("Rust memory safety", 5, 0.3, 1.0, 0.05)
    ///         .await
    ///         .unwrap();
    ///     assert!(result.steps <= 5);
    /// }
    /// ```
    pub async fn think_with(
        &mut self,
        goal: &str,
        max_iterations: usize,
        convergence_threshold: f64,
        k_proportional: f64,
        k_integral: f64,
    ) -> Result<ThinkResult> {
        self.status = Status::Thinking;

        let mut oracle = SearchOracle::new(5);
        let mut lp = ThinkLoop::new(
            goal,
            max_iterations,
            convergence_threshold,
            k_proportional,
            k_integral,
        );
        let result = lp.run(&mut oracle).await;

        if let Some(engine) = &mut self.learning_engine {
            let mut prev_state = QTable::state_key(goal);
            for signal in &result.signals {
                let next_state = QTable::state_key(&signal.observation);
                let action = engine.recommend_action(prev_state, goal, signal.step);
                engine.record_step(signal, prev_state, action, next_state);
                prev_state = next_state;
            }
            let avg_reward = if result.steps > 0 {
                result.signals.iter().map(|s| s.reward).sum::<f64>() / result.steps as f64
            } else {
                0.0
            };
            engine.end_of_episode(&lp.cold, &mut self.knowledge_index, goal, avg_reward);
        }

        for entry in lp.cold.all() {
            self.long_term_memory
                .push(Message::new("think", entry.content.clone()));
        }

        self.add_message(Message::new("think:goal", goal.to_string()));
        self.add_message(Message::new(
            "think:result",
            format!(
                "converged={} steps={} error={:.3}",
                result.converged, result.steps, result.final_error
            ),
        ));

        self.status = Status::Completed;
        Ok(result)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
