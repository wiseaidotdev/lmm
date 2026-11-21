// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::types::PipelineStage;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct LoadingIndicatorProps {
    pub stage: PipelineStage,
}

#[function_component(LoadingIndicator)]
pub fn loading_indicator(props: &LoadingIndicatorProps) -> Html {
    let label = props.stage.label();
    let icon = props.stage.icon_class();

    html! {
        <div
            class="flex items-start gap-3 mb-5 animate-fade-in"
            aria-label="Assistant is thinking"
            aria-live="polite"
            aria-atomic="true"
        >
            <div
                class="w-8 h-8 rounded-full bg-vect-accent flex items-center justify-center shrink-0 mt-1"
                aria-hidden="true"
            >
                <i class={classes!(icon, "text-white", "text-xs")} aria-hidden="true" />
            </div>
            <div class="message-ai flex items-center gap-2">
                <span class="text-vect-muted text-sm" aria-live="polite">{ label }</span>
                <span class="flex gap-1" aria-hidden="true">
                    <span
                        class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce"
                        style="animation-delay: 0ms"
                    />
                    <span
                        class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce"
                        style="animation-delay: 150ms"
                    />
                    <span
                        class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce"
                        style="animation-delay: 300ms"
                    />
                </span>
            </div>
        </div>
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
