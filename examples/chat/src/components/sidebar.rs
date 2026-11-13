use crate::types::GenerationMode;
use input_rs::yew::Input;
use wasm_bindgen::JsCast;
use web_sys::{HtmlInputElement, InputEvent, MouseEvent};
use yew::prelude::*;

fn validate_count(v: String) -> bool {
    v.parse::<usize>()
        .map(|n| (1..=200).contains(&n))
        .unwrap_or(false)
}

fn validate_float(v: String) -> bool {
    v.parse::<f64>()
        .map(|n| (0.0..=1.0).contains(&n))
        .unwrap_or(false)
}

#[derive(Properties, PartialEq)]
pub struct SidebarProps {
    pub mode: GenerationMode,
    pub on_mode_change: Callback<GenerationMode>,
    pub stochastic: bool,
    pub on_stochastic_change: Callback<bool>,
    pub probability: u32,
    pub on_probability_change: Callback<u32>,
    pub think: bool,
    pub on_think_change: Callback<bool>,
    pub deep: bool,
    pub on_deep_change: Callback<bool>,
    pub iterations_handle: UseStateHandle<String>,
    pub iterations_valid: UseStateHandle<bool>,
    pub inference_depth_handle: UseStateHandle<String>,
    pub inference_depth_valid: UseStateHandle<bool>,
    pub cf_threshold_handle: UseStateHandle<String>,
    pub cf_threshold_valid: UseStateHandle<bool>,
}

#[function_component(Sidebar)]
pub fn sidebar(props: &SidebarProps) -> Html {
    let iterations_ref = use_node_ref();
    let inference_depth_ref = use_node_ref();
    let cf_threshold_ref = use_node_ref();

    let show_tram_params = props.mode == GenerationMode::Tram;
    let show_summarize_params = props.mode == GenerationMode::Summarize;

    let on_prob = {
        let cb = props.on_probability_change.clone();
        Callback::from(move |e: InputEvent| {
            if let Some(el) = e
                .target()
                .and_then(|t| t.dyn_into::<HtmlInputElement>().ok())
                && let Ok(v) = el.value().parse::<u32>()
            {
                cb.emit(v);
            }
        })
    };

    html! {
        <nav class="flex flex-col gap-5 p-4 h-full" aria-label="VECT configuration">
            <section aria-labelledby="mode-label">
                <h2 id="mode-label" class="sidebar-label">{ "Generation Mode" }</h2>
                <div class="flex flex-col gap-0.5" role="radiogroup" aria-label="Generation mode">
                    { for GenerationMode::all().into_iter().map(|m| {
                        let active = m == props.mode;
                        let cb = props.on_mode_change.clone();
                        let mc = m.clone();
                        let onclick = Callback::from(move |_: MouseEvent| cb.emit(mc.clone()));
                        html! {
                            <button
                                key={m.label()}
                                class={classes!("mode-btn", if active { "mode-btn-active" } else { "mode-btn-inactive" })}
                                onclick={onclick}
                                role="radio"
                                aria-checked={active.to_string()}
                                id={format!("mode-{}", m.label().to_lowercase())}
                                title={m.description()}
                            >
                                <i class={classes!(m.icon_class(), "text-sm", "shrink-0", "w-4", "text-center")} aria-hidden="true"></i>
                                <div class="flex flex-col items-start min-w-0">
                                    <span class="text-sm font-medium leading-none">{m.label()}</span>
                                    <span class="text-[10px] opacity-50 leading-snug mt-0.5 truncate w-full">{m.description()}</span>
                                </div>
                            </button>
                        }
                    }) }
                </div>
            </section>
            <section aria-labelledby="stochastic-label">
                <h2 id="stochastic-label" class="sidebar-label">{ "Stochasticity" }</h2>
                <div class="flex flex-col gap-3">
                    <div class="control-row">
                        <label
                            for="stochastic-toggle"
                            class="text-sm text-vect-text cursor-pointer select-none"
                        >
                            { "Enable synonym replacement" }
                        </label>
                        <button
                            role="switch"
                            aria-checked={props.stochastic.to_string()}
                            id="stochastic-toggle"
                            onclick={{
                                let cb = props.on_stochastic_change.clone();
                                let current = props.stochastic;
                                Callback::from(move |_: MouseEvent| cb.emit(!current))
                            }}
                            class={classes!(
                                "toggle-track",
                                "shrink-0",
                                if props.stochastic { "bg-vect-accent" } else { "bg-vect-border" }
                            )}
                        >
                            <span
                                class={classes!(
                                "toggle-thumb",
                                if props.stochastic { "translate-x-5" } else { "translate-x-0" }
                            )}
                            />
                        </button>
                    </div>
                    if props.stochastic {
                        <div class="flex flex-col gap-1.5 animate-fade-in">
                            <div class="control-row">
                                <label for="prob-slider" class="text-sm text-vect-text">
                                    { "Probability" }
                                </label>
                                <span
                                    class="text-sm font-mono font-semibold text-vect-accent"
                                    aria-live="polite"
                                >
                                    { format!("{}%", props.probability) }
                                </span>
                            </div>
                            <input
                                type="range"
                                id="prob-slider"
                                min="1"
                                max="100"
                                value={props.probability.to_string()}
                                class="w-full h-1.5 rounded-full cursor-pointer accent-vect-accent bg-vect-border"
                                oninput={on_prob}
                                aria-label="Synonym replacement probability"
                                aria-valuemin="1"
                                aria-valuemax="100"
                                aria-valuenow={props.probability.to_string()}
                            />
                            <p class="text-xs text-vect-subtle">
                                { format!("{}% chance each word is replaced by a synonym.", props.probability) }
                            </p>
                        </div>
                    }
                </div>
            </section>
            if show_tram_params {
                <section aria-labelledby="tram-label">
                    <h2 id="tram-label" class="sidebar-label">{ "TRAM Parameters" }</h2>
                    <div class="flex flex-col gap-3">
                        <div class="control-row">
                            <label
                                for="think-toggle"
                                class="text-sm text-vect-text cursor-pointer select-none"
                            >
                                { "Think Loop" }
                            </label>
                            <button
                                role="switch"
                                aria-checked={props.think.to_string()}
                                id="think-toggle"
                                onclick={{
                                    let cb = props.on_think_change.clone();
                                    let current = props.think;
                                    Callback::from(move |_: MouseEvent| cb.emit(!current))
                                }}
                                class={classes!(
                                    "toggle-track",
                                    "shrink-0",
                                    if props.think { "bg-vect-accent" } else { "bg-vect-border" }
                                )}
                            >
                                <span
                                    class={classes!(
                                    "toggle-thumb",
                                    if props.think { "translate-x-5" } else { "translate-x-0" }
                                )}
                                />
                            </button>
                        </div>
                        <div class="control-row">
                            <label
                                for="deep-toggle"
                                class="text-sm text-vect-text cursor-pointer select-none"
                            >
                                { "Deep Search" }
                            </label>
                            <button
                                role="switch"
                                aria-checked={props.deep.to_string()}
                                id="deep-toggle"
                                onclick={{
                                    let cb = props.on_deep_change.clone();
                                    let current = props.deep;
                                    Callback::from(move |_: MouseEvent| cb.emit(!current))
                                }}
                                class={classes!(
                                    "toggle-track",
                                    "shrink-0",
                                    if props.deep { "bg-vect-accent" } else { "bg-vect-border" }
                                )}
                            >
                                <span
                                    class={classes!(
                                    "toggle-thumb",
                                    if props.deep { "translate-x-5" } else { "translate-x-0" }
                                )}
                                />
                            </button>
                        </div>
                        <Input
                            r#type="number"
                            label="Iterations"
                            handle={props.iterations_handle.clone()}
                            name="iterations"
                            r#ref={iterations_ref}
                            placeholder="10"
                            input_class="vect-input text-sm py-2"
                            field_class="w-full"
                            label_class="text-xs text-vect-muted block mb-1"
                            error_class="text-red-400 text-xs mt-1"
                            error_message="Enter 1 - 200"
                            valid_handle={props.iterations_valid.clone()}
                            validate_function={Callback::from(validate_count)}
                            id="iterations-input"
                        />
                        <Input
                            r#type="number"
                            label="Inference Depth"
                            handle={props.inference_depth_handle.clone()}
                            name="inference_depth"
                            r#ref={inference_depth_ref}
                            placeholder="3"
                            input_class="vect-input text-sm py-2"
                            field_class="w-full"
                            label_class="text-xs text-vect-muted block mb-1"
                            error_class="text-red-400 text-xs mt-1"
                            error_message="Enter 1 - 200"
                            valid_handle={props.inference_depth_valid.clone()}
                            validate_function={Callback::from(validate_count)}
                            id="inference-depth-input"
                        />
                        <Input
                            r#type="number"
                            label="CF Threshold"
                            handle={props.cf_threshold_handle.clone()}
                            name="cf_threshold"
                            r#ref={cf_threshold_ref}
                            placeholder="0.25"
                            input_class="vect-input text-sm py-2"
                            field_class="w-full"
                            label_class="text-xs text-vect-muted block mb-1"
                            error_class="text-red-400 text-xs mt-1"
                            error_message="Enter 0.0 - 1.0"
                            valid_handle={props.cf_threshold_valid.clone()}
                            validate_function={Callback::from(validate_float)}
                            id="cf-threshold-input"
                        />
                    </div>
                </section>
            }
            if show_summarize_params {
                <section aria-labelledby="params-label">
                    <h2 id="params-label" class="sidebar-label">{ "Summarize Parameters" }</h2>
                    <div class="flex flex-col gap-3">
                        <p class="text-xs text-vect-subtle leading-relaxed">
                            { "Summarize mode extracts key sentences from the provided text." }
                        </p>
                    </div>
                </section>
            }
            <div class="mt-auto pt-4 border-t border-vect-border/40">
                <p class="text-[11px] text-vect-subtle leading-relaxed">
                    { "Powered by " }
                    <span class="text-vect-text font-semibold">{ "LMM" }</span>
                    { " - Large Mathematical Model. All text generation runs locally in your browser via WebAssembly." }
                </p>
            </div>
        </nav>
    }
}
