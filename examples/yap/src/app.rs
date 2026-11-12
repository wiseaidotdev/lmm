use crate::components::agent_panel::AgentPanel;
use crate::components::header::Header;
use crate::lmm_bridge::tram_query;
use crate::types::{AgentId, AgentMessage, ConversationState};
use lmm::summarize_to_topic;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen_futures::spawn_local;
use web_sys::MouseEvent;
use yew::prelude::*;

struct Shared {
    running: bool,
    generation: u64,
    next_id: usize,
    domain_topic: String,
    pending_topic: String,
    alpha_messages: Vec<AgentMessage>,
    beta_messages: Vec<AgentMessage>,
}

impl Shared {
    fn new() -> Self {
        Self {
            running: false,
            generation: 0,
            next_id: 0,
            domain_topic: String::new(),
            pending_topic: String::new(),
            alpha_messages: Vec::new(),
            beta_messages: Vec::new(),
        }
    }
}

#[function_component(App)]
pub fn app() -> Html {
    let shared = use_state(|| Rc::new(RefCell::new(Shared::new())));
    let conv_state = use_state(|| ConversationState::Idle);
    let topic_handle = use_state(|| "Rust programming language".to_string());
    let topic_valid = use_state(|| true);
    let alpha_thinking = use_state(|| false);
    let beta_thinking = use_state(|| false);
    let alpha_messages_ui = use_state(Vec::<AgentMessage>::new);
    let beta_messages_ui = use_state(Vec::<AgentMessage>::new);

    let alpha_count = alpha_messages_ui.len();
    let beta_count = beta_messages_ui.len();

    let start_agent_loop = {
        let shared = Rc::clone(&*shared);
        let alpha_thinking = alpha_thinking.clone();
        let beta_thinking = beta_thinking.clone();
        let alpha_messages_ui = alpha_messages_ui.clone();
        let beta_messages_ui = beta_messages_ui.clone();

        Callback::from(move |_: ()| {
            let shared = Rc::clone(&shared);
            let alpha_thinking = alpha_thinking.clone();
            let beta_thinking = beta_thinking.clone();
            let alpha_messages_ui = alpha_messages_ui.clone();
            let beta_messages_ui = beta_messages_ui.clone();

            let generation = shared.borrow().generation;

            spawn_local(async move {
                loop {
                    let (topic, domain, alpha_turn) = {
                        let s = shared.borrow();
                        if !s.running || s.generation != generation {
                            break;
                        }
                        (
                            s.pending_topic.clone(),
                            s.domain_topic.clone(),
                            s.next_id as u64,
                        )
                    };

                    alpha_thinking.set(true);
                    let alpha_resp = tram_query(&topic, &domain, alpha_turn).await;
                    alpha_thinking.set(false);

                    let cont = {
                        let mut s = shared.borrow_mut();
                        if !s.running || s.generation != generation {
                            false
                        } else {
                            let aid = s.next_id;
                            s.next_id += 1;
                            s.alpha_messages.push(AgentMessage {
                                id: aid,
                                agent: AgentId::Alpha,
                                content: alpha_resp.clone(),
                                topic_used: topic.clone(),
                            });
                            let domain = s.domain_topic.clone();
                            let turn = s.next_id as u64;
                            s.pending_topic =
                                summarize_to_topic(&alpha_resp, &domain, &topic, turn);
                            true
                        }
                    };

                    if !cont {
                        break;
                    }

                    let alpha_all = shared.borrow().alpha_messages.clone();
                    alpha_messages_ui.set(alpha_all);

                    let (beta_topic, beta_domain, beta_turn) = {
                        let s = shared.borrow();
                        (
                            s.pending_topic.clone(),
                            s.domain_topic.clone(),
                            s.next_id as u64,
                        )
                    };

                    beta_thinking.set(true);
                    let beta_resp = tram_query(&beta_topic, &beta_domain, beta_turn).await;
                    beta_thinking.set(false);

                    let cont = {
                        let mut s = shared.borrow_mut();
                        if !s.running || s.generation != generation {
                            false
                        } else {
                            let bid = s.next_id;
                            s.next_id += 1;
                            s.beta_messages.push(AgentMessage {
                                id: bid,
                                agent: AgentId::Beta,
                                content: beta_resp.clone(),
                                topic_used: beta_topic.clone(),
                            });
                            let turn = s.next_id as u64;
                            s.pending_topic =
                                summarize_to_topic(&beta_resp, &beta_domain, &beta_topic, turn);
                            true
                        }
                    };

                    if !cont {
                        break;
                    }

                    let beta_all = shared.borrow().beta_messages.clone();
                    beta_messages_ui.set(beta_all);
                }
            });
        })
    };

    let on_start: Callback<MouseEvent> = {
        let shared = Rc::clone(&*shared);
        let conv_state = conv_state.clone();
        let topic_handle = topic_handle.clone();
        let topic_valid = topic_valid.clone();
        let alpha_messages_ui = alpha_messages_ui.clone();
        let beta_messages_ui = beta_messages_ui.clone();
        let alpha_thinking = alpha_thinking.clone();
        let beta_thinking = beta_thinking.clone();
        let start_agent_loop = start_agent_loop.clone();

        Callback::from(move |_| {
            let topic = (*topic_handle).trim().to_string();
            if topic.is_empty() {
                topic_valid.set(false);
                return;
            }
            topic_valid.set(true);

            {
                let mut s = shared.borrow_mut();
                s.running = true;
                s.generation = s.generation.wrapping_add(1);
                s.next_id = 0;
                s.domain_topic = topic.clone();
                s.pending_topic = topic;
                s.alpha_messages.clear();
                s.beta_messages.clear();
            }

            alpha_messages_ui.set(vec![]);
            beta_messages_ui.set(vec![]);
            alpha_thinking.set(false);
            beta_thinking.set(false);
            conv_state.set(ConversationState::Running);
            start_agent_loop.emit(());
        })
    };

    let on_pause: Callback<MouseEvent> = {
        let shared = Rc::clone(&*shared);
        let conv_state = conv_state.clone();

        Callback::from(move |_| {
            shared.borrow_mut().running = false;
            conv_state.set(ConversationState::Paused);
        })
    };

    let on_resume: Callback<MouseEvent> = {
        let shared = Rc::clone(&*shared);
        let conv_state = conv_state.clone();
        let start_agent_loop = start_agent_loop.clone();

        Callback::from(move |_| {
            {
                let mut s = shared.borrow_mut();
                s.running = true;
                s.generation = s.generation.wrapping_add(1);
            }
            conv_state.set(ConversationState::Running);
            start_agent_loop.emit(());
        })
    };

    let on_reset: Callback<MouseEvent> = {
        let shared = Rc::clone(&*shared);
        let conv_state = conv_state.clone();
        let alpha_messages_ui = alpha_messages_ui.clone();
        let beta_messages_ui = beta_messages_ui.clone();
        let alpha_thinking = alpha_thinking.clone();
        let beta_thinking = beta_thinking.clone();

        Callback::from(move |_| {
            {
                let mut s = shared.borrow_mut();
                s.running = false;
                s.generation = s.generation.wrapping_add(1);
                s.next_id = 0;
                s.domain_topic = String::new();
                s.pending_topic = String::new();
                s.alpha_messages.clear();
                s.beta_messages.clear();
            }

            conv_state.set(ConversationState::Idle);
            alpha_messages_ui.set(vec![]);
            beta_messages_ui.set(vec![]);
            alpha_thinking.set(false);
            beta_thinking.set(false);
        })
    };

    html! {
        <div
            class="flex flex-col h-screen overflow-hidden bg-vect-bg font-sans"
            aria-label="YAP: Two LMM Agents Yapping Non Stop"
        >
            <Header
                state={(*conv_state).clone()}
                topic_handle={topic_handle.clone()}
                topic_valid={topic_valid.clone()}
                alpha_count={alpha_count}
                beta_count={beta_count}
                on_start={on_start}
                on_pause={on_pause}
                on_resume={on_resume}
                on_reset={on_reset}
            />

            <main
                class="flex flex-1 overflow-hidden"
                id="main-content"
                aria-label="Agent conversation panels"
            >
                <AgentPanel
                    agent={AgentId::Alpha}
                    messages={(*alpha_messages_ui).clone()}
                    is_thinking={*alpha_thinking}
                    state={(*conv_state).clone()}
                />
                <AgentPanel
                    agent={AgentId::Beta}
                    messages={(*beta_messages_ui).clone()}
                    is_thinking={*beta_thinking}
                    state={(*conv_state).clone()}
                />
            </main>

            <footer
                class="px-5 py-2 border-t border-vect-border bg-vect-surface shrink-0 flex items-center justify-between"
                role="contentinfo"
            >
                <p class="text-[11px] text-vect-subtle">
                    {"Powered by "}
                    <span class="text-vect-text font-semibold">{"LMM"}</span>
                    {" · TRAM pipeline · DuckDuckGo search · WebAssembly"}
                </p>
                <p class="text-[11px] text-vect-subtle hidden sm:block">
                    {"All reasoning runs locally in your browser"}
                </p>
            </footer>
        </div>
    }
}
