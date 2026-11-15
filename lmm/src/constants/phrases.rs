// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Phrase Banks
//!
//! Rich, Band-9-calibre academic and discourse phrase banks used by
//! [`crate::generation::phrases_mixer::PhrasesMixer`].

/// High-quality academic opening sentences.
///
/// These are suitable as the first sentence of a paragraph; each ends with
/// a clause connector that invites continuation.
pub static ACADEMIC_OPENERS: &[&str] = &[
    "It is widely acknowledged that",
    "Research consistently demonstrates that",
    "A compelling argument can be made that",
    "One of the most significant aspects of this subject is",
    "When examining this topic in detail, it becomes clear that",
    "Empirical evidence strongly suggests that",
    "Scholars across disciplines have long recognised that",
    "A thorough analysis of the available evidence reveals that",
    "Contemporary understanding of this field confirms that",
    "It would be difficult to overstate the importance of the fact that",
    "Among the most noteworthy observations in this domain is that",
    "The prevailing consensus in the literature indicates that",
    "Closer inspection of the underlying principles shows that",
    "An objective evaluation of the data points to the conclusion that",
    "Critical examination of this matter consistently demonstrates that",
    "The evidence at hand makes it abundantly clear that",
    "From both a theoretical and a practical standpoint,",
    "A nuanced reading of the available material reveals that",
    "It is increasingly evident from recent studies that",
    "Upon reflection, the most substantive conclusion one can draw is that",
];

/// Discourse markers for mid-paragraph transitions.
pub static DISCOURSE_MARKERS: &[&str] = &[
    "Furthermore,",
    "In addition to this,",
    "Notably,",
    "It is equally important to note that",
    "Beyond the immediate implications,",
    "Another dimension worth considering is that",
    "Taken together, these points suggest that",
    "Of particular relevance here is the fact that",
    "This observation is further reinforced by",
    "Complementing this view,",
    "A related consideration is that",
    "What is more,",
    "By extension,",
    "In a similar vein,",
    "This point is underscored by the reality that",
    "Crucially,",
    "Significantly,",
    "Adding to this,",
    "It should also be noted that",
    "On a related note,",
];

/// Transition phrases used between paragraphs or at the start of a new idea.
pub static TRANSITION_PHRASES: &[&str] = &[
    "Building on this foundation,",
    "With this context in mind,",
    "Turning now to a broader perspective,",
    "Taking stock of the evidence so far,",
    "Moving to a related but distinct dimension,",
    "Having established this baseline,",
    "Looking at the wider picture,",
    "Bearing in mind the foregoing discussion,",
    "To appreciate the full significance of this,",
    "Shifting focus to practical implications,",
    "Against this backdrop,",
    "In light of these considerations,",
    "Proceeding from this point,",
    "Viewed through this lens,",
    "Extending this line of reasoning,",
    "In keeping with this perspective,",
    "Returning to the central question,",
    "Considering the broader ramifications,",
    "To contextualise this further,",
    "With all of this in view,",
];

/// Sentence starters for concluding statements or summary paragraphs.
pub static CONCLUSION_STARTERS: &[&str] = &[
    "In conclusion, it is evident that",
    "To summarise the key points,",
    "Ultimately, the evidence points to",
    "Drawing these threads together,",
    "In sum, the foregoing analysis demonstrates that",
    "On balance, the weight of evidence supports",
    "To bring these ideas to a close,",
    "By any reasonable measure,",
    "Taken as a whole, the available information confirms that",
    "It is therefore reasonable to conclude that",
    "All things considered,",
    "The overarching conclusion to be drawn is that",
    "In the final analysis,",
    "Synthesising the above,",
    "Looking at the totality of the evidence,",
    "It stands to reason, therefore, that",
    "The cumulative weight of these observations suggests that",
    "Rounding out this discussion,",
    "To put it plainly,",
    "The inescapable conclusion is that",
];

/// Epistemic hedges and qualifiers for measured, academic prose.
pub static ACADEMIC_HEDGES: &[&str] = &[
    "it appears that",
    "the evidence suggests that",
    "it seems reasonable to assert that",
    "one might argue that",
    "there is good reason to believe that",
    "it is plausible that",
    "the balance of evidence implies that",
    "current understanding indicates that",
    "it may be argued that",
    "it is fair to say that",
    "in many respects,",
    "to a considerable degree,",
    "for the most part,",
    "in the majority of cases,",
    "under most circumstances,",
    "by and large,",
    "as a general rule,",
    "with certain caveats,",
    "subject to context,",
    "broadly speaking,",
];

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
