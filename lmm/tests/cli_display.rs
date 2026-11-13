// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cli::display::{char_display_width, display_width};

#[test]
fn test_char_display_width_ascii() {
    assert_eq!(char_display_width('A' as u32), 1);
    assert_eq!(char_display_width('z' as u32), 1);
    assert_eq!(char_display_width('0' as u32), 1);
}

#[test]
fn test_char_display_width_cjk() {
    assert_eq!(char_display_width(0x4E00_u32), 2);
    assert_eq!(char_display_width(0x4E8C_u32), 2);
}

#[test]
fn test_char_display_width_emoji() {
    assert_eq!(char_display_width(0x1F600_u32), 2);
}

#[test]
fn test_char_display_width_zero_width() {
    assert_eq!(char_display_width(0x200B_u32), 0);
    assert_eq!(char_display_width(0xFE0F_u32), 0);
}

#[test]
fn test_display_width_ascii() {
    assert_eq!(display_width("hello"), 5);
    assert_eq!(display_width("world"), 5);
}

#[test]
fn test_display_width_mixed() {
    assert_eq!(display_width("hello world"), 11);
}

#[test]
fn test_display_width_cjk() {
    let s = "\u{4E00}\u{4E8C}";
    assert_eq!(display_width(s), 4);
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
