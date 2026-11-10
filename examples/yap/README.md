# YAP: Yapping Agents Protocol

> A Yew CSR WASM application showcasing two LMM agents conversing non-stop, powered by the full TRAM pipeline running entirely in the browser. No GPU. No API key. No training data. Just mathematics.

## What It Does

Two agents, **Agent Alpha** and **Agent Beta**, take turns querying the TRAM pipeline on a user-supplied topic. After each response, `summarize_to_topic` extracts the next sub-topic from the generated prose, so the conversation naturally pivots to related aspects of the domain over time, never repeating the same content indefinitely.

## TRAM Pipeline in WASM

Each agent turn runs the full pipeline in the browser via WebAssembly:

```
User topic
  → tram_query(topic, domain, turn)
      → build_search_query()               - strips domain key for multi-word sub-topics
                                        so Wikipedia finds different articles per turn
      → DuckDuckGo IA / CORS proxy → Wikipedia (origin=*)
      → remove_disambiguation_snippets()   - strips (band)/(film)/(song) artefacts
      → SentenceFilter                     - noise gate: metadata · low-density · <8 words
      → KnowledgeIndex (IDF-weighted ingestion)
      → kb.query() + kb.answer()
      → RuleBase.from_chunks() + InferenceEngine (depth=3, CF≥0.20)
      → ProseBuilder (slot-filling paragraph synthesis + CoherenceScorer)
      → GlobalDeduplicator                 - Jaccard dedup (threshold 0.65)
      → ParagraphShaper                    - 2 paras · 2-5 sentences · 40-180 words
      → SentenceFilter.is_acceptable()     - post-assembly quality gate
      → StochasticEnhancer (p=0.05)
  → Final response displayed
  → summarize_to_topic(response, domain, current_topic, turn)
      → next sub-topic for the other agent
```

### Sub-topic Freshness Strategy

Wikipedia's `exintro` endpoint returns the same 6 introductory sentences for any query that resolves to the same article. To prevent the agents from cycling through identical content:

- For **multi-word sub-topics** (e.g. `"Rust ownership model"`), the domain key (`"Rust"`) is stripped before the Wikipedia search, so the query becomes `"ownership model"` → resolves to the _Ownership (programming)_ article instead of the main Rust article.
- `summarize_to_topic` tracks the **current topic** and skips candidates that equal it, preventing fixed-point loops.
- Each agent maintains its own `turn` counter, feeding different seeds into the stochastic and selection layers.

## Running

```bash
cd examples/yap
trunk serve --port 3000
# Open http://localhost:3000
```

## Building for Production

```bash
trunk build
# Output in dist/
```

## License

Licensed under the [MIT License](../../LICENSE).
