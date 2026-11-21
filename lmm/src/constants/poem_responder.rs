// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Poem Responder Constants

/// Maximum number of stanzas to generate in a composed poem.
pub const POEM_MAX_STANZAS: usize = 5;

/// Number of lines per stanza.
pub const POEM_LINES_PER_STANZA: usize = 4;

/// Maximum number of words allowed in a single poem line before it is trimmed.
pub const POEM_MAX_LINE_WORDS: usize = 12;

/// Minimum number of words a corpus sentence must have to be considered as a
/// poem line candidate.
pub const POEM_MIN_SENTENCE_WORDS: usize = 4;

/// Blank line separator inserted between stanzas.
pub const POEM_STANZA_SEPARATOR: &str = "\n\n";

/// Fallback poem lines used when the KB corpus is empty.
///
/// Each element is a template with `{subject}` as a placeholder.
pub const POEM_FALLBACK_LINES: &[&str] = &[
    "In the depth of {subject}, wonder takes its form,",
    "A force that shapes the world, both still and storm,",
    "Through every age and season it endures,",
    "Its truth unchanging, timeless and pure.",
    "{subject} flows through all that ever was,",
    "A quiet presence without pause,",
    "It binds the small and great alike,",
    "A thread through life on which all things might strike.",
    "To know {subject} is to know the whole,",
    "A mirror held before the soul,",
    "What was and is and yet shall be,",
    "Reflected in its mystery.",
    "We walk beside {subject} every day,",
    "It lights the dark and clears the way,",
    "In every breath and heartbeat found,",
    "A universal, boundless sound.",
    "{subject} asks no question, needs no name,",
    "Yet every story ends the same,",
    "Return to source, return to ground,",
    "In {subject}, the lost are found.",
];

/// Prefix that indicates a retrieved/scraped page title line (not poem-worthy).
pub const POEM_JUNK_PREFIX_RETRIEVED: &str = "retrieved from";

/// Prefix pattern for search navigation artifacts.
pub const POEM_JUNK_PREFIX_SEARCH: &str = "search search";

/// Pattern for horizontal rule / separator artifacts.
pub const POEM_JUNK_PREFIX_STARS: &str = "* * *";

/// Denominator for the minimum alphabetic character ratio check on prose lines.
pub const POEM_MIN_ALPHA_RATIO_DENOMINATOR: usize = 2;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
