use crate::types::ConversationState;
use input_rs::yew::Input;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct HeaderProps {
    pub state: ConversationState,
    pub topic_handle: UseStateHandle<String>,
    pub topic_valid: UseStateHandle<bool>,
    pub alpha_count: usize,
    pub beta_count: usize,
    pub on_start: Callback<web_sys::MouseEvent>,
    pub on_pause: Callback<web_sys::MouseEvent>,
    pub on_resume: Callback<web_sys::MouseEvent>,
    pub on_reset: Callback<web_sys::MouseEvent>,
}

#[function_component(Header)]
pub fn header(props: &HeaderProps) -> Html {
    let topic_ref = use_node_ref();
    let is_idle = props.state == ConversationState::Idle;
    let is_running = props.state == ConversationState::Running;
    let is_paused = props.state == ConversationState::Paused;

    html! {
        <header
            class="flex flex-col gap-3 px-5 py-4 border-b border-vect-border bg-vect-surface shrink-0"
            role="banner"
        >
            <div class="flex items-center gap-4 flex-wrap">
                <div class="flex items-center gap-3" aria-label="YAP logo and title">
                    <div class="w-9 h-9 rounded-xl bg-vect-accent flex items-center justify-center shrink-0">
                        <i class="fa-solid fa-comments text-white text-sm" aria-hidden="true"></i>
                    </div>
                    <div>
                        <h1 class="text-lg font-bold tracking-tight text-vect-text leading-none">{"YAP: Yapping Agents Protocol"}</h1>
                        <p class="text-xs text-vect-muted leading-none mt-0.5 hidden sm:block">
                            {"Two LMM Agents Yapping Non Stop"}
                        </p>
                    </div>
                </div>

                <div class="flex items-center gap-2 ml-auto">
                    <span
                        class="badge bg-vect-elevated text-vect-muted text-xs"
                        aria-label={format!("Agent Alpha: {} messages", props.alpha_count)}
                    >
                        <i class="fa-solid fa-robot text-vect-accent text-[10px]" aria-hidden="true"></i>
                        {format!("α {}", props.alpha_count)}
                    </span>
                    <span
                        class="badge bg-vect-elevated text-vect-muted text-xs"
                        aria-label={format!("Agent Beta: {} messages", props.beta_count)}
                    >
                        <i class="fa-solid fa-brain text-purple-400 text-[10px]" aria-hidden="true"></i>
                        {format!("β {}", props.beta_count)}
                    </span>
                    <div class="w-px h-5 bg-vect-border mx-1" aria-hidden="true" />
                    <a
                        href="https://github.com/wiseaidotdev/lmm"
                        target="_blank"
                        rel="noopener noreferrer"
                        class="vect-btn-ghost h-8 w-8 rounded-lg p-0 flex items-center justify-center"
                        aria-label="LMM on GitHub"
                        title="LMM on GitHub"
                    >
                        <i class="fa-brands fa-github text-sm" aria-hidden="true"></i>
                    </a>
                    <span class="badge bg-vect-elevated text-vect-muted text-xs">{"WASM"}</span>
                    <span class="badge bg-vect-elevated text-vect-muted text-xs">{"LMM"}</span>
                </div>
            </div>

            <div
                class="flex items-center gap-3 flex-wrap"
                role="toolbar"
                aria-label="Agent conversation controls"
            >
                <div class="flex-1 min-w-[200px]" role="search" aria-label="Initial conversation topic">
                    <Input
                        r#type={"text"}
                        label={""}
                        handle={props.topic_handle.clone()}
                        name={"topic"}
                        r#ref={topic_ref}
                        placeholder={"Enter a starting topic, e.g. Rust programming language"}
                        input_class={"yap-topic-input"}
                        field_class={"w-full"}
                        error_class={"text-red-400 text-xs mt-1"}
                        error_message={"Topic cannot be empty"}
                        valid_handle={props.topic_valid.clone()}
                        validate_function={Callback::from(|v: String| !v.trim().is_empty())}
                        id={"topic-input"}
                    />
                </div>

                <div class="flex items-center gap-2 shrink-0" role="group" aria-label="Playback controls">
                    if is_idle {
                        <button
                            class="vect-btn-primary px-4 h-9 text-sm"
                            onclick={props.on_start.clone()}
                            aria-label="Start agent conversation"
                            id="btn-start"
                            type="button"
                        >
                            <i class="fa-solid fa-play text-xs" aria-hidden="true"></i>
                            {"Start"}
                        </button>
                    }

                    if is_running {
                        <button
                            class="yap-btn-pause px-4 h-9 text-sm"
                            onclick={props.on_pause.clone()}
                            aria-label="Pause agent conversation"
                            id="btn-pause"
                            type="button"
                        >
                            <i class="fa-solid fa-pause text-xs" aria-hidden="true"></i>
                            {"Pause"}
                        </button>
                    }

                    if is_paused {
                        <button
                            class="vect-btn-primary px-4 h-9 text-sm"
                            onclick={props.on_resume.clone()}
                            aria-label="Resume agent conversation"
                            id="btn-resume"
                            type="button"
                        >
                            <i class="fa-solid fa-play text-xs" aria-hidden="true"></i>
                            {"Resume"}
                        </button>
                    }

                    if !is_idle {
                        <button
                            class="yap-btn-reset px-4 h-9 text-sm"
                            onclick={props.on_reset.clone()}
                            aria-label="Reset and clear agent conversation"
                            id="btn-reset"
                            type="button"
                        >
                            <i class="fa-solid fa-rotate-left text-xs" aria-hidden="true"></i>
                            {"Reset"}
                        </button>
                    }
                </div>

                <div
                    class="flex items-center gap-2 text-xs"
                    aria-live="polite"
                    aria-label="Conversation status"
                >
                    {match &props.state {
                        ConversationState::Idle => html! {
                            <span class="flex items-center gap-1.5 text-vect-subtle">
                                <span class="w-2 h-2 rounded-full bg-vect-border inline-block" aria-hidden="true"></span>
                                {"Ready"}
                            </span>
                        },
                        ConversationState::Running => html! {
                            <span class="flex items-center gap-1.5 text-green-400">
                                <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse inline-block" aria-hidden="true"></span>
                                {"Live"}
                            </span>
                        },
                        ConversationState::Paused => html! {
                            <span class="flex items-center gap-1.5 text-yellow-400">
                                <span class="w-2 h-2 rounded-full bg-yellow-400 inline-block" aria-hidden="true"></span>
                                {"Paused"}
                            </span>
                        },
                    }}
                </div>
            </div>
        </header>
    }
}
