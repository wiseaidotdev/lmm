// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # CLI Display Utilities
//!
//! UI helper functions for formatting console output with proper Unicode width handling.

use tracing::info;

/// Returns the display width of a Unicode code point.
///
/// Certain code points (variation selectors, zero-width joiners, etc.) have
/// zero visual width. CJK characters typically have width 2.
///
/// # Arguments
///
/// * `cp` - Unicode code point as a u32.
///
/// # Time Complexity
///
/// O(1)
#[cfg(feature = "cli")]
pub fn char_display_width(cp: u32) -> usize {
    match cp {
        0xFE00..=0xFE0F | 0xE0100..=0xE01EF => 0,
        0x200B | 0x200C | 0x200D | 0xFEFF => 0,
        0x1100..=0x115F
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0x3400..=0x9FFF
        | 0xA960..=0xA97F
        | 0xAC00..=0xD7FF
        | 0xF900..=0xFAFF
        | 0xFE10..=0xFE19
        | 0xFE30..=0xFE6F
        | 0xFF01..=0xFF60
        | 0x1B000..=0x1B0FF
        | 0x1F004
        | 0x1F0CF
        | 0x1F300..=0x1FFFF
        | 0x20000..=0x2FFFD
        | 0x30000..=0x3FFFD => 2,
        _ => 1,
    }
}

/// Returns the total display width of a string.
///
/// # Arguments
///
/// * `s` - Input string.
///
/// # Time Complexity
///
/// O(|s|)
#[cfg(feature = "cli")]
pub fn display_width(s: &str) -> usize {
    s.chars().map(|c| char_display_width(c as u32)).sum()
}

/// Prints a styled banner header.
///
/// # Arguments
///
/// * `title` - Banner title text.
/// * `icon` - Emoji or icon to display before the title.
///
/// # Time Complexity
///
/// O(|title|)
#[cfg(feature = "cli")]
pub fn banner(title: &str, icon: &str) {
    let width = 54usize;
    let label = format!("  {}  {}", icon, title);
    let pad = width.saturating_sub(display_width(&label));
    info!("");
    info!("╔{}╗", "═".repeat(width));
    info!("║{}{}║", label, " ".repeat(pad));
    info!("╚{}╝", "═".repeat(width));
    info!("");
}

/// Prints a section divider with a label.
///
/// # Arguments
///
/// * `label` - Divider label text.
///
/// # Time Complexity
///
/// O(|label|)
#[cfg(feature = "cli")]
pub fn divider(label: &str) {
    let dashes = "-".repeat(46usize.saturating_sub(label.chars().count() + 4));
    info!("");
    info!("-- {} {}", label, dashes);
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
