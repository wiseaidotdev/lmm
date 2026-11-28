# 🛠️ LMM Architecture

LLM provides a variety of features that enable you to search and generate coherent English without any prior training or a GPU. All learning happens on the fly.

## 🔬 Key Capabilities

- 🛡️ **Ethical AI by design**: no training-data bias, no GPU requirement, no proprietary API dependencies. Full auditability, transparency, and explainability baked in.
- 🧬 **Genetic Programming**: population-based symbolic regression with template seeding (linear, quadratic, periodic) and variable-enforcement guards.
- 📐 **Symbolic Calculus**: automatic differentiation (chain rule, product rule, trig) and constant-folding simplification.
- 🌀 **Physics Suite**: Harmonic Oscillator, Lorenz Attractor, Pendulum, SIR Epidemic, N-body Gravity; All implement `Simulatable`.
- 🔢 **Field Calculus**: N-D gradient, Laplacian, divergence, and 3-D curl via central differences.
- 🔗 **Causal Reasoning**: structural causal models, `do(X=v)` interventions, and counterfactual queries.
- 🧩 **Neural Operators**: circular convolution with SGD kernel learning and Fourier spectral operators.
- 🔤 **Text ↔ Equation**: losslessly encode any text string into a symbolic equation and recover it exactly via integer residuals.
- 🔮 **Symbolic Prediction**: equation-native text continuation using sliding-window GP regression and vocabulary anchoring.
- 🔍 **Sentence Filter**: declarative pre-ingestion noise gate, discards metadata lines, disambiguation fragments, low information-density text, and sub-8-word snippets before they enter the KnowledgeIndex.
- 🔄 **Global Deduplicator**: Jaccard-similarity deduplication (threshold 0.65) eliminates near-duplicate sentences after expert + extractive prose is merged, ensuring clean, non-repetitive output.
- 🏗️ **Paragraph Shaper**: structural output formatter enforcing 2 paragraphs, 2-5 sentences per paragraph, and 40-180 words per paragraph with discourse connector injection between paragraphs.
- 🎲 **Stochastic Enhancement**: synonym-bank word replacement (`--stochastic`) delivers unique output each run while preserving mathematical sentence structure.
- 🎨 **Spectral Image Synthesis**: generate procedural PPM images from a text prompt by hashing it into Fourier wave components.
- 🧠 **Expert System**: IF-THEN production rules auto-derived from KB chunks at runtime; backward-chaining inference engine with Prospector-model certainty factors; how/why explanation facility; slot-filling `ProseBuilder` that constructs novel paragraphs, never copies verbatim.
- 🔀 **Coherence Scoring**: greedy nearest-neighbour sentence reordering by token overlap maximises topical flow across generated paragraphs.
- 💡 **Adaptive Thought Engine (ATE)**: multi-step symbolic reasoner that classifies each prompt into one of eight reasoning modes (`factual`, `causal`, `procedural`, `analogical`, `definitional`, `exploratory`, `evaluative`, `task`) and iterates through typed thought steps to build a goal-directed reasoning chain before the main pipeline runs. Activated with `--think` / `-t`.
- 🔁 **Recursive Belief Propagation Loop (`--loop N`)**: re-runs the full TRAM pipeline up to N times. After each pass a [`BeliefNet`](lmm/src/cognition/bayesian/) is built from the top KB chunks and prompt-evidence is propagated to yield a posterior belief interval. A scalar quality score (computed by `rewards::compute_response_reward`) measures answer improvement. The loop exits early when the quality delta falls below `--loop-threshold` (default `0.05`). The best-scoring pass wins. Combining with `--think` and `--deep` enables iterative retrieval refinement across passes.
- 🎯 **Context-Aware Response Framing**: the final answer is prefixed with an intent-specific phrase (e.g. "The advantages of Rust are..."). Always-on; suppressible with `--no-frame`.
- 🔍 **Task Decomposer**: fires up to `--max-tasks` intent-specific web queries to enrich the KB with mode-relevant content before reasoning.
- 💻 **Code Responder**: for `task`-mode prompts, renders a fenced code block with templates or KB-extracted snippets.

## ⚙️ Architecture

```mermaid
flowchart TD
    A["User: lmm -p 'topic'"] --> B["handle_tram()"]
    B --> WEB["Web Prefetch\nSearchAggregator\n(DuckDuckGo IA → CORS proxy → Wikipedia)"]
    WEB --> DEAMB["remove_disambiguation_snippets()\nStrip (band) / (film) / (song) artefacts"]
    DEAMB --> SFILT["SentenceFilter\nPre-ingestion noise gate\n(metadata · low-density · too-short · bare-question-titles)"]
    SFILT --> IDX["KnowledgeIndex\n(IDF-weighted ingestion)"]

    B --> TDEC["TaskDecomposer\nClassify prompt mode\nFire intent-specific sub-queries\n(advantages of · vs · how to · code example)"]
    TDEC --> IDX

    B --> LOOP_CHECK{"--loop N > 1?"}
    LOOP_CHECK -- yes --> LOOP_ORCH["LoopOrchestrator\nrun_loop(pass=1..N)"]
    LOOP_ORCH --> LOOP_PASS["Per-pass: full pipeline\n(knowledge → thinking → retrieval → assembly)"]
    LOOP_PASS --> BNET["BeliefNet.from_chunks()\nBayesian belief propagation"]
    BNET --> BPOST["posterior = propagate(prompt_tokens)\nbelief interval [lo, mid, hi]"]
    BPOST --> QUAL["rewards::compute_response_reward()\nquality score ∈ [0, 1]"]
    QUAL --> CONV{"delta < loop-threshold\nor pass == N?"}
    CONV -- no --> LOOP_PASS
    CONV -- yes --> BEST["Return best-scoring pass answer"]
    BEST --> IDX
    LOOP_CHECK -- no --> ATE_CHECK

    B --> ATE_CHECK{"--think flag?"}
    ATE_CHECK -- yes --> ATE["Adaptive Thought Engine\nClassify prompt mode\n(factual/causal/procedural/analogical/\ndefinitional/exploratory/evaluative/task)"]
    ATE --> ATE_STEPS["Typed Thought Steps\n(deductive · abductive · analogical · corrective)\nSearchOracle queries per sub-goal"]
    ATE_STEPS --> ATE_CHAIN["ThoughtChain\nfactual_insight → KB seed\nprioritized_queries → ThinkLoop goal"]
    ATE_CHAIN --> TL
    ATE_CHECK -- no --> TL

    B --> TL["ThinkLoop.run()\nPID-style closed-loop reasoning"]
    TL --> TL1["Reflector.formulate_query()"]
    TL1 --> TL2["SearchOracle.fetch()"]
    TL2 --> TL3["CognitionSignal: Jaccard error"]
    TL3 --> TL4{"error < 0.70?"}
    TL4 -- yes --> TL5["converged = true"]
    TL4 -- no --> TL6{"observation empty?"}
    TL6 -- yes --> TL7["stall_streak++"]
    TL6 -- no --> TL8["stall_streak = 0"]
    TL7 & TL8 --> TL1
    TL5 --> COLD["ColdStore: memory chunks → KB"]
    COLD --> IDX

    IDX --> KA["KnowledgeIndex.answer(prompt, top_k)"]
    KA --> RULES["RuleBase.from_chunks()\nIF-THEN production rules"]
    RULES --> INFER["InferenceEngine\nBackward chaining · max_depth=3"]
    INFER --> CF{"CF ≥ 0.20?"}
    CF -- yes --> PROSE["ProseBuilder\nSlot-filling paragraph synthesis\n+ CoherenceScorer"]
    CF -- no --> PROSE_SKIP["Skip expert path"]

    KA --> MRG["Merge expert + extractive prose"]
    PROSE --> MRG
    PROSE_SKIP --> MRG

    MRG --> DEDUP["GlobalDeduplicator\nJaccard similarity dedup\n(threshold = 0.65)"]
    DEDUP --> SFILT2["SentenceFilter.is_acceptable()\nPost-merge quality gate\n+ dangling-preposition filter\n+ bare-question-title filter"]
    SFILT2 --> OPENER["is_good_opener()\nBAD_OPENERS + CONTINUATION_ADVERBS check\nSwap best opener to position 0"]
    OPENER --> SHAPER["ParagraphShaper\n2 paragraphs · 2-5 sentences each\n40-180 words per paragraph"]
    SHAPER --> STOCH["StochasticEnhancer\nSynonym substitution (p = 0.05)"]
    STOCH --> FRAME["ResponseFramer\nIntent-specific opening phrase\n(evaluative/causal/analogical/procedural/task)"]
    FRAME --> OUT["Final Output\ncoherent · intent-aware · unique · bias-free"]
```
