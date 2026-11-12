// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Encode/Decode Command Handlers
//!
//! Handlers for encoding and decoding operations.

use crate::{
    cli::display,
    cli::util,
    encode::{decode_from_parts, decode_message, encode_text},
    error::Result as LmmResult,
};
use tracing::info;

pub fn handle_encode(
    input: String,
    text: String,
    iterations: usize,
    depth: usize,
    verbose: bool,
) -> LmmResult<()> {
    display::banner("Encode · GP Symbolic Compression", "📐");
    let source = util::load_source(&input, text)?;
    info!("  📝 Input   : {:?}", source);
    info!("  📏 Length  : {} chars", source.len());
    let encoded = encode_text(&source, iterations, depth)?;
    let decoded = decode_message(&encoded)?;
    if verbose {
        display::divider("Equation");
        info!("  {}", encoded.summary());
        display::divider("Encoded Data");
        info!("  {}", encoded.to_data_string());
        display::divider("Round-trip Verify");
        let status = if decoded == source {
            "✅ PERFECT"
        } else {
            "⚠️  lossy (residuals correct)"
        };
        info!("  {}", status);
        info!("  Decoded : {:?}", decoded);
        info!("");
        info!("  💾 To decode later:");
        info!(
            "     lmm decode --equation {:?} --length {} --residuals {:?}",
            encoded.equation.to_string(),
            encoded.length,
            encoded
                .residuals
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    } else {
        println!(
            "{} {} {}",
            encoded.equation,
            encoded.length,
            encoded
                .residuals
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    Ok(())
}

pub fn handle_decode(
    equation: String,
    length: usize,
    residuals: String,
    verbose: bool,
) -> LmmResult<()> {
    display::banner("Decode · Equation to Text", "🔓");
    info!("  📐 Equation : {}", equation);
    info!("  📏 Length   : {}", length);
    let decoded = decode_from_parts(&equation, length, &residuals)?;
    if verbose {
        display::divider("Decoded Text");
        info!("  {}", decoded);
    } else {
        println!("{}", decoded);
    }
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
