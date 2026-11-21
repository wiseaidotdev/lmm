// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::types::PromptCard;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct PromptCardProps {
    pub card: PromptCard,
    pub on_click: Callback<String>,
}

#[function_component(PromptCardItem)]
pub fn prompt_card_item(props: &PromptCardProps) -> Html {
    let description = props.card.description.to_string();
    let on_click = props.on_click.clone();

    let onclick = Callback::from(move |_: web_sys::MouseEvent| {
        on_click.emit(description.clone());
    });

    let description_preview: String = props.card.description.chars().take(60).collect();

    html! {
        <button
            class="bg-vect-surface border border-vect-border rounded-xl p-4 text-left \
                   hover:border-vect-accent hover:bg-vect-elevated \
                   transition-all duration-150 cursor-pointer focus:outline-none \
                   focus:ring-2 focus:ring-vect-accent focus:ring-offset-2 \
                   focus:ring-offset-vect-bg w-full"
            onclick={onclick}
            aria-label={format!("Send prompt: {}", props.card.description)}
            type="button"
        >
            <i class={classes!(props.card.icon, "text-lg", "text-vect-accent")} aria-hidden="true" />
            <p class="text-sm font-semibold text-vect-text mt-2">{ props.card.title }</p>
            <p class="text-xs text-vect-muted mt-1 leading-snug">{ description_preview }</p>
        </button>
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
