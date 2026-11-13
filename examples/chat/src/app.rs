use crate::components::chat_window::ChatWindow;
use crate::components::header::Header;
use crate::components::input_bar::InputBar;
use crate::components::sidebar::Sidebar;
use crate::lmm_bridge;
use crate::types::{ChatMessage, GenerationMode, MessageRole};
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

#[function_component(App)]
pub fn app() -> Html {
    let messages = use_state(Vec::<ChatMessage>::new);
    let mode = use_state(|| GenerationMode::Tram);
    let stochastic = use_state(|| false);
    let probability = use_state(|| 30u32);
    let think = use_state(|| false);
    let deep = use_state(|| false);
    let iterations_handle = use_state(|| "10".to_string());
    let iterations_valid = use_state(|| true);
    let inference_depth_handle = use_state(|| "3".to_string());
    let inference_depth_valid = use_state(|| true);
    let cf_threshold_handle = use_state(|| "0.25".to_string());
    let cf_threshold_valid = use_state(|| true);
    let is_loading = use_state(|| false);
    let next_id = use_state(|| 0usize);
    let sidebar_open = use_state(|| false);

    let on_mode_change = {
        let mode = mode.clone();
        Callback::from(move |m: GenerationMode| mode.set(m))
    };
    let on_stochastic_change = {
        let s = stochastic.clone();
        Callback::from(move |v: bool| s.set(v))
    };
    let on_probability_change = {
        let p = probability.clone();
        Callback::from(move |v: u32| p.set(v))
    };
    let on_think_change = {
        let t = think.clone();
        Callback::from(move |v: bool| t.set(v))
    };
    let on_deep_change = {
        let d = deep.clone();
        Callback::from(move |v: bool| d.set(v))
    };

    let on_submit = {
        let messages = messages.clone();
        let mode = mode.clone();
        let stochastic = stochastic.clone();
        let probability = probability.clone();
        let think = think.clone();
        let deep = deep.clone();
        let iter_h = iterations_handle.clone();
        let depth_h = inference_depth_handle.clone();
        let cf_h = cf_threshold_handle.clone();
        let is_loading = is_loading.clone();
        let next_id = next_id.clone();

        Callback::from(move |text: String| {
            let id = *next_id;
            next_id.set(id + 2);

            let stoch = *stochastic;
            let prob = *probability as f64 / 100.0;
            let thnk = *think;
            let dp = *deep;
            let iter = (*iter_h).parse::<usize>().unwrap_or(10);
            let depth = (*depth_h).parse::<usize>().unwrap_or(3);
            let cf = (*cf_h).parse::<f64>().unwrap_or(0.25);

            let user_msg = ChatMessage {
                id,
                role: MessageRole::User,
                content: text.clone(),
                mode: None,
                timestamp: String::new(),
                links: vec![],
            };

            let current_mode = (*mode).clone();

            let mut msgs = (*messages).clone();
            msgs.push(user_msg);
            messages.set(msgs.clone());
            is_loading.set(true);
            let messages = messages.clone();
            let is_loading = is_loading.clone();
            spawn_local(async move {
                let (response, links) = match current_mode {
                    GenerationMode::Tram => {
                        lmm_bridge::tram_query(&text, thnk, dp, iter, depth, cf, stoch, prob).await
                    }
                    GenerationMode::Summarize => {
                        let resp = lmm_bridge::summarize(&text, 3, stoch, prob);
                        (resp, vec![])
                    }
                };
                let mut m = msgs;
                m.push(ChatMessage {
                    id: id + 1,
                    role: MessageRole::Assistant,
                    content: response,
                    mode: Some(current_mode),
                    timestamp: String::new(),
                    links,
                });
                messages.set(m);
                is_loading.set(false);
            });
        })
    };

    let sidebar_content = html! {
        <Sidebar
            mode={(*mode).clone()}
            on_mode_change={on_mode_change.clone()}
            stochastic={*stochastic}
            on_stochastic_change={on_stochastic_change.clone()}
            probability={*probability}
            on_probability_change={on_probability_change.clone()}
            think={*think}
            on_think_change={on_think_change.clone()}
            deep={*deep}
            on_deep_change={on_deep_change.clone()}
            iterations_handle={iterations_handle.clone()}
            iterations_valid={iterations_valid.clone()}
            inference_depth_handle={inference_depth_handle.clone()}
            inference_depth_valid={inference_depth_valid.clone()}
            cf_threshold_handle={cf_threshold_handle.clone()}
            cf_threshold_valid={cf_threshold_valid.clone()}
        />
    };

    html! {
        <div class="flex flex-col h-screen overflow-hidden bg-vect-bg font-sans">
            <Header />
            <div class="flex flex-1 overflow-hidden">
                <aside
                    class="hidden lg:flex flex-col w-72 border-r border-vect-border bg-vect-surface overflow-y-auto shrink-0"
                    aria-label="LMM CI configuration panel"
                >
                    { sidebar_content.clone() }
                </aside>
                if *sidebar_open {
                    <div
                        class="lg:hidden fixed inset-0 z-50 flex"
                        role="dialog"
                        aria-modal="true"
                        aria-label="Configuration panel"
                    >
                        <aside
                            class="w-72 bg-vect-surface border-r border-vect-border overflow-y-auto"
                        >
                            { sidebar_content }
                        </aside>
                        <div
                            class="flex-1 bg-black/60 cursor-pointer"
                            onclick={{
                                let so = sidebar_open.clone();
                                Callback::from(move |_| so.set(false))
                            }}
                        />
                    </div>
                }
                <main class="flex flex-col flex-1 overflow-hidden" id="main-content">
                    <div class="lg:hidden px-4 pt-3 shrink-0">
                        <button
                            class="vect-btn-ghost text-sm"
                            onclick={{
                                let so = sidebar_open.clone();
                                Callback::from(move |_| so.set(!*so))
                            }}
                            aria-label="Toggle configuration panel"
                            aria-expanded={(*sidebar_open).to_string()}
                        >
                            <i class="fa-solid fa-sliders w-4 h-4" aria-hidden="true" />
                            { "Configure" }
                        </button>
                    </div>
                    <ChatWindow messages={(*messages).clone()} is_loading={*is_loading} />
                    <InputBar on_submit={on_submit} is_loading={*is_loading} />
                </main>
            </div>
        </div>
    }
}
