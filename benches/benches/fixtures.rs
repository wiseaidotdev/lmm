// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
//! Shared canned-data fixtures for TRAM benchmarks.

/// Corpus about Rust for `KnowledgeIndex` benchmarks.
pub static RUST_CORPUS: &[&str] = &[
    "Rust is a systems programming language focused on safety, speed, and concurrency.",
    "The borrow checker enforces memory safety at compile time without a garbage collector.",
    "Rust uses an ownership model to manage memory, eliminating entire classes of bugs.",
    "Zero-cost abstractions allow high-level code that compiles to efficient machine code.",
    "Rust prevents data races at compile time through its type system and ownership rules.",
    "The Rust toolchain includes Cargo, a powerful package manager and build system.",
    "Rustfmt and Clippy provide automated formatting and linting for Rust projects.",
    "Rust's standard library provides traits, collections, and concurrency primitives.",
    "WebAssembly support allows Rust code to run in web browsers at near-native speed.",
    "Async/await syntax in Rust enables efficient asynchronous programming without overhead.",
    "Rust has been voted the most loved programming language in Stack Overflow surveys for years.",
    "Memory safety in Rust is guaranteed without a runtime cost, unlike garbage-collected languages.",
    "The enum type in Rust enables algebraic data types and exhaustive pattern matching.",
    "Rust's trait system provides polymorphism similar to interfaces in other languages.",
    "Error handling in Rust uses the Result type, making failures explicit and recoverable.",
    "Rust enables fearless concurrency by enforcing thread safety through the type system.",
    "The Rust compiler produces extremely informative error messages to guide developers.",
    "Lifetimes in Rust annotate how long references are valid, preventing dangling pointers.",
    "Rust's performance is comparable to C and C++ while providing stronger safety guarantees.",
    "Foreign function interface allows Rust to call C libraries and be called from other languages.",
    "Rust macros enable compile-time code generation and domain-specific languages.",
    "The standard library's HashMap provides efficient key-value storage with O(1) average lookup.",
    "Iterators in Rust are lazy and compose efficiently through adapter chains.",
    "Rust's type inference reduces verbosity while maintaining strong static typing.",
    "Smart pointers like Box, Rc, and Arc provide flexible memory management patterns.",
    "Rust's module system enables clear code organisation and controlled visibility.",
    "The cargo test framework integrates unit tests, integration tests, and documentation tests.",
    "Rust generics enable code reuse without runtime overhead through monomorphisation.",
    "Pattern matching in Rust is exhaustive, ensuring all cases are handled at compile time.",
    "Rust's unsafe blocks allow low-level operations while making unsafety explicit and auditable.",
    "The tokio runtime provides async I/O, timers, and task scheduling for Rust programs.",
    "Rust's string types distinguish between owned Strings and borrowed string slices.",
    "Closures in Rust capture their environment and can be passed as first-class values.",
    "Rust's const and static keywords support compile-time evaluation and global constants.",
    "The serde library provides fast, compile-time-generated serialisation for Rust types.",
    "Rust's build scripts allow custom code generation and system integration at build time.",
    "Conditional compilation with cfg attributes allows platform-specific code in Rust.",
    "Rust supports no_std environments for embedded systems programming without an OS.",
    "The Pin type in Rust enables self-referential structures needed for async state machines.",
    "Rust's documentation system generates HTML docs from inline doc comments.",
    "Rayon enables data-parallelism in Rust through simple iterator-based parallel computation.",
    "Rust's allocation control lets programs use custom allocators for specialised workloads.",
    "The From and Into traits provide ergonomic type conversion in Rust programs.",
    "Rust's Display and Debug traits control how values are printed to output.",
    "Channels in Rust provide message-passing concurrency between threads and async tasks.",
    "Rust's atomic types enable lock-free data structures and low-level synchronisation.",
    "The standard library's Vec provides a growable array with amortised O(1) push.",
    "Rust's BTreeMap provides sorted key-value storage with O(log n) operations.",
    "Associated types in Rust traits reduce type parameter clutter in generic interfaces.",
    "Rust's where clauses express complex trait bounds clearly in generic function signatures.",
];

/// Corpus about Python for variety.
pub static PYTHON_CORPUS: &[&str] = &[
    "Python is a high-level, general-purpose programming language known for readability.",
    "Python uses dynamic typing, allowing variables to hold values of any type at runtime.",
    "The Python Package Index hosts hundreds of thousands of third-party libraries.",
    "Python's indentation-based syntax enforces code structure without braces.",
    "List comprehensions in Python provide concise syntax for creating and transforming lists.",
    "Python supports multiple programming paradigms including procedural, object-oriented, and functional.",
    "The CPython interpreter executes Python code via a bytecode virtual machine.",
    "NumPy provides efficient multi-dimensional array operations for scientific computing in Python.",
    "Pandas offers high-performance data manipulation and analysis for tabular data.",
    "Python's Global Interpreter Lock prevents true thread parallelism in CPython.",
    "Asyncio provides event-loop-based asynchronous programming for I/O-bound tasks.",
    "Type hints in Python allow optional static type checking with tools like mypy.",
    "Python's duck typing favours interface over inheritance for polymorphism.",
    "The with statement enables clean resource management through context managers.",
    "Python generators produce values lazily, enabling memory-efficient iteration.",
    "Decorators in Python wrap functions to add behaviour without modifying their source.",
    "Python's exception handling uses try, except, else, and finally blocks.",
    "The itertools module provides efficient iterators for combinatorial operations.",
    "Python dictionaries provide O(1) average key-value lookup using hash tables.",
    "F-strings allow inline expression interpolation in Python string literals.",
];

/// Corpus about Water for the multi-topic e2e benchmark.
pub static WATER_CORPUS: &[&str] = &[
    "Water is a chemical compound consisting of two hydrogen atoms bonded to one oxygen atom.",
    "Water covers approximately 71 percent of Earth's surface as oceans, lakes, and rivers.",
    "The molecular formula of water is H2O with a molecular weight of 18.015 g/mol.",
    "Water has an unusually high heat capacity, stabilising Earth's climate and temperature.",
    "Liquid water reaches its maximum density at approximately 4 degrees Celsius.",
    "Water is an excellent solvent, dissolving more substances than almost any other liquid.",
    "The hydrogen bonds between water molecules give water its distinctive surface tension.",
    "Water is essential for all known forms of life on Earth.",
    "Photosynthesis uses water molecules as an electron donor to produce oxygen.",
    "The water cycle describes the continuous movement of water through Earth's atmosphere.",
    "Capillary action allows water to flow upward in narrow tubes against gravity.",
    "Water has a high latent heat of vaporisation, making it efficient for cooling.",
    "Seawater contains approximately 3.5 percent dissolved salts, primarily sodium chloride.",
    "Fresh water represents only about 2.5 percent of all water on Earth.",
    "Ice is less dense than liquid water, causing it to float on the surface of lakes.",
    "Water plays a critical role in metabolic reactions within living cells.",
    "The triple point of water occurs at 273.16 K and 611.657 Pa.",
    "Water vapour is a significant greenhouse gas in Earth's atmosphere.",
    "Ultrapure water used in semiconductor manufacturing must be nearly free of ions.",
    "Electrolysis of water produces hydrogen and oxygen gas using electrical current.",
];

/// A multi-step proof trace for `ProseBuilder` benchmarks.
/// Each element is `(conclusion, certainty)`.
pub static PROOF_STEPS_5: &[(&str, f64)] = &[
    ("rust is memory-safe", 0.95),
    ("rust enables zero-cost abstractions", 0.88),
    ("rust prevents data races", 0.91),
    ("rust provides fearless concurrency", 0.85),
    ("rust compiles to native machine code", 0.97),
];

pub static PROOF_STEPS_1: &[(&str, f64)] = &[("rust is memory-safe", 0.95)];

/// Corpus as a single joined string for ingest benchmarks.
pub fn rust_corpus_joined(n: usize) -> String {
    RUST_CORPUS
        .iter()
        .take(n)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn python_corpus_joined() -> String {
    PYTHON_CORPUS.join(" ")
}

pub fn water_corpus_joined() -> String {
    WATER_CORPUS.join(" ")
}

/// Returns `n` sentences as a Vec<String>.
pub fn rust_sentences(n: usize) -> Vec<String> {
    RUST_CORPUS
        .iter()
        .cycle()
        .take(n)
        .map(|s| s.to_string())
        .collect()
}

/// A realistic mix of valid and junk sentences for the sentence filter benchmark.
pub static JUNK_SENTENCES: &[&str] = &[
    "Category: Programming languages",
    "See also: Garbage collection",
    "References",
    "External links",
    "^ Cite error",
    "Retrieved 2023-01-01.",
    "This article is a stub.",
    "Rust version 1.70 released.",
    "by.",
    "in.",
    ", ,",
    "1.",
    "2.",
    "§",
    "Navigation menu",
    "Jump to navigation",
    "Jump to search",
    "Edit source",
    "View history",
    "Talk page",
];

/// Prompt strings for e2e pipeline benchmarks.
pub static BENCH_PROMPTS: &[&str] = &[
    "what are the advantages of rust?",
    "what is python programming language?",
    "what are the properties of water?",
    "how does rust handle memory safety?",
    "what are the benefits of systems programming?",
];
