// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::components::loading_indicator::LoadingIndicator;
use crate::components::message_bubble::MessageBubble;
use crate::components::prompt_card::PromptCardItem;
use crate::types::{ChatMessage, PipelineStage, PromptCard};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct ChatWindowProps {
    pub messages: Vec<ChatMessage>,
    pub is_loading: bool,
    pub loading_stage: PipelineStage,
    pub on_card_click: Callback<String>,
}

#[function_component(ChatWindow)]
pub fn chat_window(props: &ChatWindowProps) -> Html {
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

    html! {
        <section
            class="flex-1 overflow-y-auto px-4 md:px-8 py-6"
            aria-label="Chat conversation"
            id="chat-window"
        >
            if props.messages.is_empty() {
                <div
                    class="flex flex-col items-center justify-center h-full gap-8 text-center max-w-2xl mx-auto"
                >
                    <div>
                        <div
                            class="w-14 h-14 rounded-2xl bg-vect-elevated flex items-center justify-center mx-auto mb-4"
                        >
                            <i
                                class="fa-solid fa-bolt text-vect-accent text-2xl"
                                aria-hidden="true"
                            />
                        </div>
                        <h2 class="text-2xl font-bold text-vect-text mb-2">
                            { "Welcome to LMM CI" }
                        </h2>
                        <p class="text-vect-muted text-sm leading-relaxed">
                            { "Choose a prompt below to get started, or type your own." }
                        </p>
                    </div>
                    <div
                        class="grid grid-cols-1 md:grid-cols-3 gap-3 w-full"
                        role="list"
                        aria-label="Suggested prompts"
                    >
                        { for PromptCard::templates().into_iter().map(|card| {
                            let on_click = props.on_card_click.clone();
                            html! {
                                <div key={card.title} role="listitem">
                                    <PromptCardItem card={card} on_click={on_click} />
                                </div>
                            }
                        }) }
                    </div>
                </div>
            } else {
                <div class="flex flex-col max-w-4xl mx-auto">
                    { for props.messages.iter().map(|msg| html! {
                        <MessageBubble key={msg.id} message={msg.clone()} />
                    }) }
                    if props.is_loading {
                        <LoadingIndicator stage={props.loading_stage.clone()} />
                    }
                    <div ref={bottom_ref} id="chat-bottom" aria-hidden="true" />
                </div>
            }
        </section>
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
