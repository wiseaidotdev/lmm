use crate::types::{AgentId, AgentMessage};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct MessageBubbleProps {
    pub message: AgentMessage,
}

#[function_component(MessageBubble)]
pub fn message_bubble(props: &MessageBubbleProps) -> Html {
    let is_alpha = props.message.agent == AgentId::Alpha;
    let bubble_cls = props.message.agent.bubble_class();
    let icon_cls = props.message.agent.icon_class();
    let accent_cls = props.message.agent.accent_class();
    let label = props.message.agent.label();

    let avatar_cls =
        format!("w-7 h-7 rounded-full flex items-center justify-center shrink-0 {accent_cls}");
    let icon_full_cls = format!("{icon_cls} text-white text-[10px]");
    let bubble_full_cls = format!("{bubble_cls} flex flex-col gap-1 max-w-full flex-1");

    html! {
        <article
            class="flex items-end gap-2 mb-4 animate-fade-in w-full"
            aria-label={format!("Message from {label}")}
            aria-live="polite"
        >
            if is_alpha {
                <div class={avatar_cls.clone()} aria-hidden="true">
                    <i class={icon_full_cls.clone()} aria-hidden="true"></i>
                </div>
            }
            <div class={bubble_full_cls}>
                <span class="text-[10px] font-semibold opacity-60 leading-none px-1">
                    {label}
                    {" · "}
                    <span class="font-normal opacity-80">{&props.message.topic_used}</span>
                </span>
                <p class="text-sm leading-relaxed whitespace-pre-wrap break-words">
                    {&props.message.content}
                </p>
            </div>
            if !is_alpha {
                <div class={avatar_cls} aria-hidden="true">
                    <i class={icon_full_cls} aria-hidden="true"></i>
                </div>
            }
        </article>
    }
}
