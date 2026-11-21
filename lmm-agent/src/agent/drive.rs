// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Internal drive methods for [`LmmAgent`].
//!
//! Covers [`drive_state`](LmmAgent::drive_state),
//! [`record_residual`](LmmAgent::record_residual),
//! [`record_incoherence`](LmmAgent::record_incoherence), and
//! [`record_contradiction`](LmmAgent::record_contradiction).

use lmm::cognition::drive::DriveState;

use super::LmmAgent;

impl LmmAgent {
    /// Emits the current [`DriveState`] by ticking the agent's [`InternalDrive`].
    ///
    /// If no drive has been accumulated via [`LmmAgent::record_residual`] the
    /// returned state will be idle.  The drive counters are reset after each
    /// call, matching the semantics of [`InternalDrive::tick`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    ///
    /// let mut agent = LmmAgent::new("Curious".into(), "Learn everything.".into());
    /// agent.record_residual(0.9);
    /// let state = agent.drive_state();
    /// assert!(!state.signals.is_empty());
    /// ```
    pub fn drive_state(&mut self) -> DriveState {
        self.internal_drive.tick()
    }

    /// Feeds an unexplained prediction residual into the agent's internal drive.
    ///
    /// Calling this after each world-model error accumulates curiosity that
    /// surfaces on the next [`drive_state`](Self::drive_state) call.
    pub fn record_residual(&mut self, magnitude: f64) {
        self.internal_drive.record_residual(magnitude);
    }

    /// Feeds an incoherence signal into the agent's internal drive.
    pub fn record_incoherence(&mut self, magnitude: f64) {
        self.internal_drive.record_incoherence(magnitude);
    }

    /// Notifies the drive system that a contradiction was detected in memory.
    pub fn record_contradiction(&mut self) {
        self.internal_drive.record_contradiction();
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
