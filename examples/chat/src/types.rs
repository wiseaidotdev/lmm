// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationMode {
    Tram,
    Summarize,
}

impl GenerationMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Tram => "LMM",
            Self::Summarize => "Summarize",
        }
    }

    pub fn icon_class(&self) -> &'static str {
        match self {
            Self::Tram => "fa-solid fa-brain",
            Self::Summarize => "fa-solid fa-scissors",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Tram => "Text Reasoning and Applied Mathematics pipeline",
            Self::Summarize => "Extract key sentences from a given corpus",
        }
    }

    pub fn color_class(&self) -> &'static str {
        match self {
            Self::Tram => "bg-vect-accent/20 text-vect-accent",
            Self::Summarize => "bg-vect-elevated text-vect-text",
        }
    }

    pub fn all() -> Vec<GenerationMode> {
        vec![GenerationMode::Tram, GenerationMode::Summarize]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchLink {
    pub title: String,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodeBlock {
    pub language: String,
    pub code: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub id: usize,
    pub role: MessageRole,
    pub content: String,
    pub code_blocks: Vec<CodeBlock>,
    pub mode: Option<GenerationMode>,
    pub timestamp: String,
    pub links: Vec<SearchLink>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UploadedFile {
    pub name: String,
    pub text: String,
    pub byte_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptCard {
    pub title: &'static str,
    pub description: &'static str,
    pub icon: &'static str,
}

impl PromptCard {
    pub fn templates() -> Vec<PromptCard> {
        vec![
            PromptCard {
                title: "Text Generation",
                description: "What are the advantages of rust?",
                icon: "fa-solid fa-align-left",
            },
            PromptCard {
                title: "Code Generation",
                description: "write a simple rust hello world program.",
                icon: "fa-solid fa-code",
            },
        ]
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    Idle,
    SearchingWeb,
    DeepSearch,
    AdaptiveThoughtEngine,
    ThinkLoopReasoning,
    KnowledgeRetrieval,
    ExpertInference,
    Composing,
}

impl PipelineStage {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Idle => "",
            Self::SearchingWeb => "Searching Web...",
            Self::DeepSearch => "Deep Scraping...",
            Self::AdaptiveThoughtEngine => "Thinking...",
            Self::ThinkLoopReasoning => "Reasoning...",
            Self::KnowledgeRetrieval => "Knowledge Retrieval...",
            Self::ExpertInference => "Expert Inference...",
            Self::Composing => "Composing response...",
        }
    }

    pub fn icon_class(&self) -> &'static str {
        match self {
            Self::Idle => "fa-solid fa-bolt",
            Self::SearchingWeb => "fa-solid fa-globe",
            Self::DeepSearch => "fa-solid fa-magnifying-glass",
            Self::AdaptiveThoughtEngine => "fa-solid fa-brain",
            Self::ThinkLoopReasoning => "fa-solid fa-rotate",
            Self::KnowledgeRetrieval => "fa-solid fa-book-open",
            Self::ExpertInference => "fa-solid fa-diagram-project",
            Self::Composing => "fa-solid fa-pen-nib",
        }
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to those terms.
