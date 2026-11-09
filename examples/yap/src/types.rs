#[derive(Debug, Clone, PartialEq)]
pub enum AgentId {
    Alpha,
    Beta,
}

impl AgentId {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Alpha => "Agent Alpha",
            Self::Beta => "Agent Beta",
        }
    }

    pub fn icon_class(&self) -> &'static str {
        match self {
            Self::Alpha => "fa-solid fa-robot",
            Self::Beta => "fa-solid fa-brain",
        }
    }

    pub fn accent_class(&self) -> &'static str {
        match self {
            Self::Alpha => "bg-vect-accent",
            Self::Beta => "bg-yap-beta",
        }
    }

    pub fn bubble_class(&self) -> &'static str {
        match self {
            Self::Alpha => "message-alpha",
            Self::Beta => "message-beta",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentMessage {
    pub id: usize,
    pub agent: AgentId,
    pub content: String,
    pub topic_used: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConversationState {
    Idle,
    Running,
    Paused,
}
