use crate::components::message_bubble::MessageBubble;
use crate::types::{AgentId, AgentMessage, ConversationState};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct AgentPanelProps {
    pub agent: AgentId,
    pub messages: Vec<AgentMessage>,
    pub is_thinking: bool,
    pub state: ConversationState,
}

#[function_component(AgentPanel)]
pub fn agent_panel(props: &AgentPanelProps) -> Html {
    let bottom_ref = use_node_ref();

    {
        let bottom_ref = bottom_ref.clone();
        let len = props.messages.len();
        use_effect_with(len, move |_| {
            if let Some(el) = bottom_ref.cast::<web_sys::Element>() {
                el.scroll_into_view();
            }
            move || {}
        });
    }

    let is_alpha = props.agent == AgentId::Alpha;
    let label = props.agent.label();
    let icon_cls = props.agent.icon_class();
    let accent_cls = props.agent.accent_class();

    let panel_id = if is_alpha {
        "panel-alpha"
    } else {
        "panel-beta"
    };
    let heading_id = if is_alpha {
        "heading-alpha"
    } else {
        "heading-beta"
    };

    let header_cls = if is_alpha {
        "flex items-center gap-2 px-4 py-2.5 border-b border-r border-vect-border shrink-0"
    } else {
        "flex items-center gap-2 px-4 py-2.5 border-b border-vect-border shrink-0"
    };

    let scroll_cls = if is_alpha {
        "flex-1 overflow-y-auto px-3 py-4 border-r border-vect-border"
    } else {
        "flex-1 overflow-y-auto px-3 py-4"
    };

    let icon_badge_cls =
        format!("w-6 h-6 rounded-lg flex items-center justify-center {accent_cls}");
    let icon_sm_cls = format!("{icon_cls} text-white text-[10px]");
    let idle_avatar_cls =
        format!("w-12 h-12 rounded-2xl flex items-center justify-center {accent_cls}");
    let idle_icon_cls = format!("{icon_cls} text-white text-xl");
    let thinking_avatar_cls =
        format!("w-7 h-7 rounded-full flex items-center justify-center shrink-0 {accent_cls}");
    let thinking_icon_cls = format!("{icon_cls} text-white text-[10px]");

    html! {
        <section
            class="flex flex-col flex-1 min-w-0 border-vect-border overflow-hidden"
            aria-labelledby={heading_id}
            id={panel_id}
        >
            <div class={header_cls}>
                <div class={icon_badge_cls} aria-hidden="true">
                    <i class={icon_sm_cls} aria-hidden="true"></i>
                </div>
                <h2
                    class="text-sm font-semibold text-vect-text"
                    id={heading_id}
                >
                    {label}
                </h2>
                <span
                    class="ml-auto text-xs text-vect-subtle"
                    aria-label={format!("{} message count", label)}
                >
                    {format!("{} msg", props.messages.len())}
                </span>
            </div>

            <div
                class={scroll_cls}
                role="log"
                aria-label={format!("{} conversation", label)}
                aria-live="polite"
            >
                if props.messages.is_empty() && props.state == ConversationState::Idle {
                    <div class="flex flex-col items-center justify-center h-full gap-4 text-center px-4">
                        <div class={idle_avatar_cls} aria-hidden="true">
                            <i class={idle_icon_cls} aria-hidden="true"></i>
                        </div>
                        <div>
                            <p class="text-sm font-semibold text-vect-text">{label}</p>
                            <p class="text-xs text-vect-muted mt-1 leading-relaxed">
                                {"Waiting to start. Enter a topic above and press Start."}
                            </p>
                        </div>
                    </div>
                }

                {for props.messages.iter().map(|msg| html! {
                    <MessageBubble key={msg.id} message={msg.clone()} />
                })}

                if props.is_thinking {
                    <div
                        class="flex items-center gap-2 mb-4 animate-fade-in"
                        aria-label={format!("{} is thinking", label)}
                        aria-live="polite"
                    >
                        <div class={thinking_avatar_cls} aria-hidden="true">
                            <i class={thinking_icon_cls} aria-hidden="true"></i>
                        </div>
                        <div class="flex items-center gap-2 px-3 py-2 rounded-xl bg-vect-elevated">
                            <span class="text-xs text-vect-muted">{"Searching & reasoning"}</span>
                            <span class="flex gap-1" aria-hidden="true">
                                <span class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce" style="animation-delay: 0ms"></span>
                                <span class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce" style="animation-delay: 150ms"></span>
                                <span class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce" style="animation-delay: 300ms"></span>
                            </span>
                        </div>
                    </div>
                }

                <div ref={bottom_ref} aria-hidden="true" />
            </div>
        </section>
    }
}
