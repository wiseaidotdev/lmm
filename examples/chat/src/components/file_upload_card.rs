// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::types::UploadedFile;
use input_rs::yew::Input;
use js_sys::{ArrayBuffer, Promise, Reflect};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{DragEvent, Event, FileList, FileReader, HtmlInputElement, MouseEvent, Window};
use yew::prelude::*;

fn is_supported_file(name: &str, mime: &str) -> bool {
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let supported_exts = [
        "txt", "md", "markdown", "rs", "py", "js", "ts", "html", "htm", "css", "json", "toml",
        "yaml", "yml", "csv", "xml", "sh", "c", "cpp", "h", "java", "go", "rb", "php", "swift",
        "kt", "pdf",
    ];
    supported_exts.contains(&ext.as_str()) || mime.starts_with("text/")
}

fn is_pdf(name: &str, mime: &str) -> bool {
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    ext == "pdf" || mime == "application/pdf"
}

fn call_extract_pdf_text(array_buffer: &ArrayBuffer) -> Result<JsFuture, JsValue> {
    let window: Window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let fn_val = Reflect::get(&window, &JsValue::from_str("extractPdfText"))?;
    let fn_obj = fn_val
        .dyn_ref::<js_sys::Function>()
        .ok_or_else(|| JsValue::from_str("extractPdfText is not a function"))?;
    let promise_val = fn_obj.call1(&JsValue::NULL, array_buffer)?;
    let promise = promise_val
        .dyn_into::<Promise>()
        .map_err(|_| JsValue::from_str("extractPdfText did not return a Promise"))?;
    Ok(JsFuture::from(promise))
}

fn read_file_list(list: FileList, cb: Callback<Vec<UploadedFile>>) {
    for i in 0..list.length() {
        if let Some(file) = list.item(i) {
            let name = file.name();
            let mime = file.type_();
            let size = file.size() as usize;
            let cb_clone = cb.clone();
            let name_clone = name.clone();

            if !is_supported_file(&name, &mime) {
                continue;
            }

            if is_pdf(&name, &mime) {
                let reader = match FileReader::new() {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let reader_ref = reader.clone();
                let onload = Closure::wrap(Box::new(move |_: Event| {
                    let reader_inner = reader_ref.clone();
                    let cb_inner = cb_clone.clone();
                    let name_inner = name_clone.clone();
                    spawn_local(async move {
                        let result = reader_inner.result();
                        if let Ok(val) = result {
                            if let Some(ab) = val.dyn_ref::<ArrayBuffer>() {
                                match call_extract_pdf_text(ab) {
                                    Ok(future) => {
                                        if let Ok(text_val) = future.await {
                                            let text = text_val.as_string().unwrap_or_default();
                                            if !text.trim().is_empty() {
                                                cb_inner.emit(vec![UploadedFile {
                                                    name: name_inner,
                                                    byte_size: size,
                                                    text,
                                                }]);
                                            }
                                        }
                                    }
                                    Err(_) => {}
                                }
                            }
                        }
                    });
                }) as Box<dyn FnMut(_)>);

                reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                onload.forget();
                reader.read_as_array_buffer(&file).ok();
            } else {
                let reader = match FileReader::new() {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let reader_ref = reader.clone();
                let onload = Closure::wrap(Box::new(move |_: Event| {
                    if let Ok(val) = reader_ref.result() {
                        let text = val.as_string().unwrap_or_default();
                        if !text.trim().is_empty() {
                            cb_clone.emit(vec![UploadedFile {
                                name: name_clone.clone(),
                                byte_size: size,
                                text,
                            }]);
                        }
                    }
                }) as Box<dyn FnMut(_)>);

                reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                onload.forget();
                reader.read_as_text(&file).ok();
            }
        }
    }
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[derive(Properties, PartialEq)]
pub struct FileUploadCardProps {
    pub uploaded_files: Vec<UploadedFile>,
    pub on_files: Callback<Vec<UploadedFile>>,
    pub on_remove: Callback<usize>,
}

#[function_component(FileUploadCard)]
pub fn file_upload_card(props: &FileUploadCardProps) -> Html {
    let drag_active = use_state(|| false);
    let file_input_ref = use_node_ref();
    let file_input_handle = use_state(String::default);
    let file_input_valid = use_state(|| true);

    let on_dragover = {
        let drag = drag_active.clone();
        Callback::from(move |e: DragEvent| {
            e.prevent_default();
            drag.set(true);
        })
    };

    let on_dragleave = {
        let drag = drag_active.clone();
        Callback::from(move |_: DragEvent| drag.set(false))
    };

    let on_drop = {
        let drag = drag_active.clone();
        let on_files = props.on_files.clone();
        Callback::from(move |e: DragEvent| {
            e.prevent_default();
            drag.set(false);
            if let Some(dt) = e.data_transfer() {
                if let Some(list) = dt.files() {
                    read_file_list(list, on_files.clone());
                }
            }
        })
    };

    let on_change = {
        let on_files = props.on_files.clone();
        let file_ref = file_input_ref.clone();
        Callback::from(move |_: Event| {
            if let Some(el) = file_ref.cast::<HtmlInputElement>() {
                if let Some(list) = el.files() {
                    read_file_list(list, on_files.clone());
                }
            }
        })
    };

    let drop_border = if *drag_active {
        "border-vect-accent bg-vect-accent/5"
    } else {
        "border-vect-border bg-vect-elevated/40 hover:border-vect-accent/60 hover:bg-vect-elevated/70"
    };

    let zone_class = format!(
        "relative flex flex-col items-center justify-center gap-3 rounded-xl \
         border-2 border-dashed p-8 text-center transition-all duration-200 {drop_border}"
    );

    html! {
        <article
            aria-labelledby="file-card-heading"
            class="bg-vect-surface border border-vect-border rounded-xl p-5 flex flex-col gap-4"
        >
            <header class="flex items-center gap-2">
                <div
                    class="w-8 h-8 rounded-lg bg-vect-accent/15 flex items-center justify-center shrink-0"
                >
                    <i class="fa-solid fa-file-lines text-vect-accent text-sm" aria-hidden="true" />
                </div>
                <div>
                    <h3
                        id="file-card-heading"
                        class="text-sm font-semibold text-vect-text leading-none"
                    >
                        { "Explore Document" }
                    </h3>
                </div>
            </header>
            <div
                class={zone_class}
                ondragover={on_dragover}
                ondragleave={on_dragleave}
                ondrop={on_drop}
                role="region"
                aria-label="File drop zone"
            >
                <i class="fa-solid fa-cloud-arrow-up text-2xl text-vect-muted" aria-hidden="true" />
                <div>
                    <p class="text-sm font-medium text-vect-text">
                        { "Drop files here or " }
                        <label
                            for="file-upload-input"
                            class="text-vect-accent underline underline-offset-2 cursor-pointer \
                                   focus-within:ring-2 focus-within:ring-vect-accent focus-within:rounded"
                        >
                            { "browse" }
                        </label>
                    </p>
                </div>
                <div onchange={on_change} class="hidden" aria-hidden="true">
                    <Input
                        r#type="file"
                        label=""
                        handle={file_input_handle}
                        name="file-upload"
                        r#ref={file_input_ref}
                        input_class="sr-only"
                        field_class=""
                        label_class=""
                        error_class=""
                        valid_handle={file_input_valid}
                        validate_function={Callback::from(|_: String| true)}
                        id="file-upload-input"
                        aria_label="Select files to upload for document exploration"
                    />
                </div>
            </div>
            if !props.uploaded_files.is_empty() {
                <section aria-label="Uploaded files">
                    <h4
                        class="text-xs font-semibold uppercase tracking-widest text-vect-subtle mb-2"
                    >
                        { format!("{} file(s) loaded into knowledge base", props.uploaded_files.len()) }
                    </h4>
                    <ul class="flex flex-col gap-1" role="list">
                        { for props.uploaded_files.iter().enumerate().map(|(i, f)| {
                            let on_remove = props.on_remove.clone();
                            let onclick_remove = Callback::from(move |_: MouseEvent| on_remove.emit(i));
                            let icon = if f.name.ends_with(".pdf") {
                                "fa-solid fa-file-pdf text-red-400"
                            } else {
                                "fa-solid fa-file-code text-vect-accent"
                            };
                            html! {
                                <li
                                    key={f.name.clone()}
                                    class="flex items-center justify-between gap-2 px-3 py-2 \
                                           rounded-lg bg-vect-elevated border border-vect-border/60"
                                    role="listitem"
                                >
                                    <div class="flex items-center gap-2 min-w-0">
                                        <i
                                            class={format!("{icon} text-xs shrink-0")}
                                            aria-hidden="true"
                                        />
                                        <span
                                            class="text-xs text-vect-text truncate"
                                            title={f.name.clone()}
                                        >
                                            { &f.name }
                                        </span>
                                    </div>
                                    <div class="flex items-center gap-2 shrink-0">
                                        <span class="text-[10px] text-vect-subtle">
                                            { format_size(f.byte_size) }
                                        </span>
                                        <button
                                            type="button"
                                            onclick={onclick_remove}
                                            class="text-vect-muted hover:text-red-400 transition-colors \
                                                   focus:outline-none focus:ring-2 focus:ring-vect-accent \
                                                   rounded p-0.5"
                                            aria-label={format!("Remove file {}", f.name)}
                                        >
                                            <i class="fa-solid fa-xmark text-xs" aria-hidden="true" />
                                        </button>
                                    </div>
                                </li>
                            }
                        }) }
                    </ul>
                    <p class="text-xs text-vect-muted mt-2 leading-snug">
                        { "Now type your question in the input below to explore these documents." }
                    </p>
                </section>
            }
        </article>
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
