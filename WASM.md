# LMM WebAssembly Guide 🌐

LMM natively targets `wasm32-unknown-unknown` and runs the full **TRAM** (Text Reasoning and Applied Mathematics) pipeline entirely in the browser, no server, no GPU, no API key.

## Supported Frameworks

LMM WASM integration is actively used and tested with:

- **[Yew](https://yew.rs/)**: the most popular Rust / WASM frontend framework
- **[Dioxus](https://dioxuslabs.com/)**: cross-platform Rust UI framework
- **[Leptos](https://leptos.dev/)**: fine-grained reactive framework

## 📦 Adding to a WASM Project

Add `lmm` to your `Cargo.toml` with the appropriate feature set. Avoid enabling `rust-binary`, `python`, or `node` on WASM targets.

```toml
[dependencies]
lmm = { version = "0.3.7", default-features = false, features = ["wasm-net"] }
```

Build with Trunk (Yew / Leptos):

```sh
trunk serve
```

## 🌍 CORS Considerations

LMM's WASM networking layer chains multiple data sources with graceful fallback:

1. **DuckDuckGo Instant Answer API**: returns a structured abstract for most topics.
1. **CORS proxy → Wikipedia** (`origin=*`): full article text fetched via a configurable proxy when DuckDuckGo returns insufficient content.
1. **Wikipedia direct** (`origin=*`): fallback when no proxy is configured.

DuckDuckGo's endpoints set permissive CORS headers on most responses, but behaviour can vary by endpoint and region. If you encounter CORS errors:

- Use a server-side proxy to forward requests from your own origin (configure via `DDG_CORS_PROXY_URL` in `.cargo/config.toml`).
- Use the `wt-wt` (worldwide) region, which tends to have the broadest CORS coverage.
- Consider caching results server-side and serving them from your own API.

## 🔬 Feature Flags for WASM

| Feature       | WASM-safe | Description                                     |
| ------------- | --------- | ----------------------------------------------- |
| (default)     | ✅        | Core symbolic engine (no networking)            |
| `wasm-net`    | ✅        | Enables `reqwest` fetch-based HTTP on WASM      |
| `net`         | ❌        | Native Tokio networking: incompatible with WASM |
| `rust-binary` | ❌        | CLI binary: incompatible with WASM              |
| `python`      | ❌        | pyo3 bindings: incompatible with WASM           |
| `node`        | ❌        | napi bindings: incompatible with WASM           |

## 📖 Example: Yew Component

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
