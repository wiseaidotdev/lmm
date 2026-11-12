// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # CLI Command Handlers
//!
//! Individual handler modules for each CLI command.

pub mod ask;
pub mod causal;
pub mod consciousness;
pub mod discover;
pub mod encode;
pub mod field;
pub mod generate;
pub mod imagen;
pub mod physics;
pub mod simulate;

pub use causal::handle_causal;
pub use consciousness::handle_consciousness;
pub use discover::handle_discover;
pub use encode::{handle_decode, handle_encode};
pub use physics::handle_physics;
pub use simulate::handle_simulate;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
