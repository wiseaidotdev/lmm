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
pub struct ChatMessage {
    pub id: usize,
    pub role: MessageRole,
    pub content: String,
    pub mode: Option<GenerationMode>,
    pub timestamp: String,
    pub links: Vec<SearchLink>,
}
