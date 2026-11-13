// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to these terms.

//! # Simulation Constants

pub const RK4_HALF: f64 = 0.5;
pub const RK4_SIXTH: f64 = 1.0 / 6.0;
pub const RK4_WEIGHT_MIDDLE: f64 = 2.0;

pub const RK45_C2: f64 = 1.0 / 4.0;
pub const RK45_A31: f64 = 3.0 / 32.0;
pub const RK45_A32: f64 = 9.0 / 32.0;
pub const RK45_A41: f64 = 1932.0 / 2197.0;
pub const RK45_A42: f64 = -7200.0 / 2197.0;
pub const RK45_A43: f64 = 7296.0 / 2197.0;
pub const RK45_B4_1: f64 = 25.0 / 216.0;
pub const RK45_B4_3: f64 = 1408.0 / 2565.0;
pub const RK45_B4_4: f64 = 2197.0 / 4104.0;
pub const RK45_B5_1: f64 = 16.0 / 135.0;
pub const RK45_B5_3: f64 = 6656.0 / 12825.0;
pub const RK45_B5_4: f64 = 28561.0 / 56430.0;
pub const RK45_SAFETY: f64 = 0.9;
pub const RK45_EXPONENT: f64 = 0.2;
pub const RK45_STEP_SHRINK: f64 = 0.1;
pub const RK45_STEP_GROW: f64 = 5.0;
pub const RK45_DOUBLE: f64 = 2.0;
