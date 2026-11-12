// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(feature = "cli")]
mod tests {
    use lmm::cli::tram::response::{is_answer_intent_relevant, is_quality_answer};
    use lmm::cognition::thought::classifier::ReasoningMode;

    #[test]
    fn test_is_quality_answer_too_short() {
        assert!(!is_quality_answer("short text"));
    }

    #[test]
    fn test_is_quality_answer_lack_variety() {
        let answer =
            "Word word word word word. Word word word word word. Word word word word word.";
        assert!(!is_quality_answer(answer));
    }

    #[test]
    fn test_is_answer_intent_relevant_evaluative_advantages() {
        let answer = "Rust has many advantages including performance and memory safety.";
        assert!(is_answer_intent_relevant(
            answer,
            ReasoningMode::Evaluative,
            "what are the advantages of rust"
        ));
    }

    #[test]
    fn test_is_answer_intent_relevant_evaluative_disadvantages() {
        let answer = "Rust has disadvantages such as a steep learning curve.";
        assert!(is_answer_intent_relevant(
            answer,
            ReasoningMode::Evaluative,
            "what are the disadvantages of rust"
        ));
    }

    #[test]
    fn test_is_answer_intent_relevant_causal() {
        let answer = "Rust does not have a garbage collector because it uses ownership.";
        assert!(is_answer_intent_relevant(
            answer,
            ReasoningMode::Causal,
            "why does rust not have a garbage collector"
        ));
    }

    #[test]
    fn test_is_answer_intent_relevant_analogical() {
        let answer = "Rust and Python are both programming languages with different strengths.";
        assert!(is_answer_intent_relevant(
            answer,
            ReasoningMode::Analogical,
            "rust vs python"
        ));
    }

    #[test]
    fn test_is_answer_intent_relevant_missing_keywords() {
        let answer = "This is a generic answer with no relevant keywords.";
        assert!(!is_answer_intent_relevant(
            answer,
            ReasoningMode::Evaluative,
            "what are the advantages of rust"
        ));
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
