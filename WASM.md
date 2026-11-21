# LMM WebAssembly Guide 🌐

LMM natively targets `wasm32-unknown-unknown` and runs the full **TRAM** (Text Reasoning and Applied Mathematics) pipeline entirely in the browser, no server, no GPU, no API key.

## Supported Frameworks

LMM WASM integration is actively used and tested with:

- **[Yew](https://yew.rs/)**: the most popular Rust / WASM frontend framework
- **[Dioxus](https://dioxuslabs.com/)**: cross-platform Rust UI framework
- **[Leptos](https://leptos.dev/)**: fine-grained reactive framework

## 📦 Adding to a WASM Project

Add `lmm` to your `Cargo.toml` with the **`net`** feature. The single `net` feature is both WASM-safe and native-safe, `#[cfg(target_arch = "wasm32")]` gates select the correct backend automatically.

```toml
[dependencies]
lmm = { version = "0.4.3", default-features = false, features = ["net"] }
```

> [!IMPORTANT]
> Use `features = ["net"]`, **not** `"wasm-net"`. The separate `wasm-net` feature has been retired; `net` now handles both native and WASM targets via compile-time gates.

Build and serve with Trunk (Yew / Leptos):

```sh
trunk serve
```

### Programming Language Disambiguation

For evaluative queries such as _"What are the advantages of Rust?"_, the WASM pipeline detects known PL names (from `KNOWN_PROGRAMMING_LANGUAGES`) and appends `" programming language"` to the DDG query so that `"advantages of rust programming language"` resolves to the PL article rather than the material.

## 🌍 CORS Considerations

DuckDuckGo and Wikipedia both set permissive CORS headers, but behaviour can vary by endpoint and region. If you encounter CORS errors:

- Use the `wt-wt` (worldwide) region for DuckDuckGo Lite requests.
- Wikipedia's mobile-sections API (`/api/rest_v1/`) sends `Access-Control-Allow-Origin: *` and requires no proxy.
- For DuckDuckGo, consider a server-side proxy if your deployment region has CORS restrictions.

## 🔬 Feature Flags for WASM

| Feature       | WASM-safe | Description                                                       |
| ------------- | --------- | ----------------------------------------------------------------- |
| (default)     | ✅        | Core symbolic engine (no networking)                              |
| `net`         | ✅        | Unified networking, `gloo-net` on WASM, `reqwest`/tokio on native |
| `cli`         | ✅        | CLI helpers (not a binary, WASM-safe as a library dependency)     |
| `rust-binary` | ❌        | Standalone `lmm` CLI binary, incompatible with WASM               |
| `python`      | ❌        | pyo3 bindings, incompatible with WASM                             |
| `node`        | ❌        | napi-derive bindings, incompatible with WASM                      |

## 📖 Example: Yew TRAM Chat Component

```rust , ignore
use lmm::cognition::knowledge::KnowledgeIndex;
use lmm::tram::{TramConfig, TramPipeline};
use yew::prelude::*;

#[function_component(TramChat)]
fn tram_chat() -> Html {
    let response = use_state(String::new);

    let onsubmit = {
        let response = response.clone();
        Callback::from(move |prompt: String| {
            let response = response.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let config = TramConfig::builder().think(true).deep(true).build();
                let mut index = KnowledgeIndex::new();
                let pipeline = TramPipeline::new(config);
                if let Ok(result) = pipeline.run(&prompt, &mut index).await {
                    response.set(result.final_text);
                }
            });
        })
    };

    html! {
        <div>
            <p>{ (*response).clone() }</p>
        </div>
    }
}
```

## 📖 Example: Yew TextPredictor Component

```rust
use lmm::prelude::*;
use yew::prelude::*;

#[function_component(PredictorDemo)]
fn predictor_demo() -> Html {
    let output = use_state(|| String::from("Click to generate..."));
    let onclick = {
        let output = output.clone();
        Callback::from(move |_| {
            let predictor = TextPredictor::new(20, 30, 3);
            if let Ok(result) = predictor.predict_continuation("Wise AI built the first LMM", 80) {
                output.set(result.continuation);
            }
        })
    };
    html! {
        <div>
            <button {onclick}>{ "Generate" }</button>
            <p>{ (*output).clone() }</p>
        </div>
    }
}
```

## 📄 License

Licensed under the [MIT License](LICENSE).
