// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

//! # Imagen Constants

pub const LCG_MULTIPLIER: u64 = 6364136223846793005;
pub const LCG_INCREMENT: u64 = 1442695040888963407;
pub const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
pub const FNV_PRIME: u64 = 1099511628211;
pub const PROMPT_INDEX_MULTIPLIER: u64 = 31;

pub const WAVE_BAND_STRIDE: u64 = 997;
pub const WAVE_COMPONENT_STRIDE: u64 = 31337;
pub const MIN_WAVE_COMPONENTS: usize = 3;
pub const MAX_WAVE_COMPONENTS: usize = 32;
pub const FRACTAL_MAX_ITERATIONS: u32 = 32;
pub const FRACTAL_ESCAPE_RADIUS_SQ: f64 = 4.0;
pub const STYLE_SIGMA_SEED_OFFSET: u64 = 99;
pub const FRACTAL_C_REAL_SEED_OFFSET: u64 = 77;
pub const FRACTAL_C_IMAG_SEED_OFFSET: u64 = 78;
