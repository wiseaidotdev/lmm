// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::knowledge::KnowledgeIndex;
use lmm::expert::inference::InferenceEngine;
use lmm::expert::rule::RuleBase;
use lmm::generation::dedup::GlobalDeduplicator;
use lmm::generation::filter::SentenceFilter;
use lmm::generation::prose::ProseBuilder;
use lmm::generation::shaper::ParagraphShaper;
use lmm::net::{SearchAggregator, corpus_from_response};
use lmm::stochastic::StochasticEnhancer;
use lmm::text::TextSummarizer;

const TOP_K: usize = 8;
const INFERENCE_DEPTH: usize = 3;
const CF_THRESHOLD: f64 = 0.20;
const STOCHASTIC_PROB: f64 = 0.05;
const MIN_KB_WORDS: usize = 15;

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

fn topic_seed(topic: &str, turn: u64) -> u64 {
    fnv1a(topic)
        .wrapping_add(turn.wrapping_mul(2_654_435_761))
        .rotate_left(17)
}

fn is_tech_domain(domain: &str) -> bool {
    const TECH_SIGNALS: &[&str] = &[
        "programming",
        "language",
        "algorithm",
        "software",
        "compiler",
        "framework",
        "library",
        "database",
        "runtime",
        "protocol",
        "rust",
        "python",
        "java",
        "javascript",
        "typescript",
        "golang",
        "swift",
        "kotlin",
        "scala",
        "haskell",
        "lisp",
        "functional",
        "object-oriented",
        "systems",
        "embedded",
        "async",
        "concurrent",
        "machine learning",
        "deep learning",
        "neural",
        "blockchain",
        "kubernetes",
        "docker",
        "linux",
        "unix",
        "assembly",
        "llvm",
    ];
    let lo = domain.to_lowercase();
    TECH_SIGNALS.iter().any(|&s| lo.contains(s))
}

const PROGRAMMING_ASPECTS: &[&str] = &[
    "ownership model",
    "pattern matching",
    "type system",
    "memory safety",
    "concurrency model",
    "error handling",
    "trait objects",
    "generic programming",
    "functional programming",
    "async programming",
    "borrow checker",
    "package manager",
    "standard library",
    "systems programming",
    "compiler design",
    "zero-cost abstractions",
    "algebraic data types",
    "higher-order functions",
    "memory management",
    "embedded systems",
];

const GENERAL_ASPECTS: &[&str] = &[
    "history",
    "culture",
    "geography",
    "people",
    "economy",
    "architecture",
    "religion",
    "politics",
    "society",
    "notable",
    "overview",
    "characteristics",
    "development",
    "influence",
    "significance",
];

fn domain_key(domain: &str) -> &str {
    domain
        .split_whitespace()
        .find(|w| {
            let lo = w.to_lowercase();
            w.len() >= 4
                && !STOPWORDS.contains(&lo.as_str())
                && !FALSE_TOPIC_WORDS.contains(&lo.as_str())
        })
        .unwrap_or_else(|| domain.split_whitespace().next().unwrap_or("knowledge"))
}

fn aspect_topic(domain: &str, turn: u64) -> String {
    let key = domain_key(domain);
    if is_tech_domain(domain) {
        let aspect = PROGRAMMING_ASPECTS[(turn as usize) % PROGRAMMING_ASPECTS.len()];
        format!("{} {}", key, aspect)
    } else {
        let aspect = GENERAL_ASPECTS[(turn as usize) % GENERAL_ASPECTS.len()];
        format!("{} {}", key, aspect)
    }
}

fn build_search_query(topic: &str, domain: &str, turn: u64) -> String {
    if turn <= 1 {
        return topic.to_string();
    }
    let key = domain_key(domain).to_lowercase();
    let lower_topic = topic.to_lowercase();
    if lower_topic.starts_with(&key) && topic.len() > key.len() + 1 {
        let concept = topic[key.len()..].trim();
        if concept.split_whitespace().count() >= 2 {
            return concept.to_string();
        }
    }
    topic.to_string()
}

fn remove_disambiguation_snippets(corpus: &str) -> String {
    const DISAMBIG_MARKERS: &[&str] = &[
        "(band)",
        "(film)",
        "(song)",
        "(album)",
        "(TV series)",
        "(TV show)",
        "(game)",
        "(novel)",
        "(book)",
        "(comics)",
        "(character)",
        "(video game)",
        "(disambiguation)",
    ];
    corpus
        .split(". ")
        .filter(|s| !DISAMBIG_MARKERS.iter().any(|&m| s.contains(m)))
        .collect::<Vec<_>>()
        .join(". ")
}

fn strip_discourse_prefix(s: &str) -> String {
    const CONNECTORS: &[&str] = &[
        "Furthermore, ",
        "In addition, ",
        "Notably, ",
        "Moreover, ",
        "As a result, ",
        "In practice, ",
        "Interestingly, ",
        "Consequently, ",
        "Building on this, ",
        "In summary, ",
        "To summarize, ",
        "In conclusion, ",
        "Broadly speaking, ",
        "By design, ",
        "In essence, ",
        "In technical terms, ",
    ];
    for c in CONNECTORS {
        if let Some(rest) = s.strip_prefix(c) {
            let trimmed = rest.trim_start();
            if trimmed.is_empty() {
                continue;
            }
            let mut out = trimmed.to_string();
            if let Some(first) = out.get_mut(0..1) {
                first.make_ascii_uppercase();
            }
            return out;
        }
    }
    s.to_string()
}

fn is_on_topic(sentence: &str, topic_word: &str) -> bool {
    if topic_word.is_empty() {
        return true;
    }
    let first = match sentence.split_whitespace().next() {
        Some(w) => w,
        None => return true,
    };
    if first.len() < 4 || !first.chars().all(|c| c.is_alphabetic()) {
        return true;
    }
    let mut chars = first.chars();
    let starts_upper = chars.next().map(|c| c.is_uppercase()).unwrap_or(false);
    let rest_lower = chars.all(|c| c.is_lowercase());
    if !starts_upper || !rest_lower {
        return true;
    }
    let first_lower = first.to_lowercase();
    if first_lower.contains(topic_word) || topic_word.contains(&first_lower) {
        return true;
    }
    sentence.to_lowercase().contains(topic_word)
}

fn assemble(text: &str, topic: &str, seed: u64) -> String {
    let topic_word: String = topic
        .split_whitespace()
        .find(|w| {
            w.len() > 3
                && !matches!(
                    *w,
                    "what"
                        | "does"
                        | "how"
                        | "the"
                        | "and"
                        | "for"
                        | "with"
                        | "about"
                        | "that"
                        | "this"
                        | "from"
                        | "into"
                        | "over"
                )
        })
        .unwrap_or("")
        .to_lowercase();

    let sentences: Vec<String> = text
        .split('\n')
        .flat_map(|line| {
            line.split(". ")
                .map(|s| s.trim().trim_end_matches('.').trim().to_string())
                .filter(|s| s.split_whitespace().count() >= 6)
        })
        .map(|s| strip_discourse_prefix(&s))
        .filter(|s| SentenceFilter::is_acceptable(s))
        .filter(|s| is_on_topic(s, &topic_word))
        .collect();

    if sentences.is_empty() {
        return text.to_string();
    }

    let mut shuffled = sentences;
    let n = shuffled.len();
    let mut rng = seed;
    for i in (1..n).rev() {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (rng >> 33) as usize % (i + 1);
        shuffled.swap(i, j);
    }

    let deduplicator = GlobalDeduplicator::new(0.65);
    let mut unique = deduplicator.deduplicate(shuffled);

    if unique.is_empty() {
        return text.to_string();
    }

    if let Some(pos) = unique
        .iter()
        .position(|s| SentenceFilter::is_good_opener(s))
    {
        unique.swap(0, pos);
    }

    ParagraphShaper::new(2, 2, 5, 40, 180).shape(&unique, seed)
}

pub async fn tram_query(topic: &str, domain: &str, turn: u64) -> String {
    let topic = topic.trim();
    let seed = topic_seed(topic, turn);

    let search_query = build_search_query(topic, domain, turn);

    let aggregator = SearchAggregator::new();

    let raw_a = aggregator
        .get_response(&search_query)
        .await
        .map(|r| corpus_from_response(&r))
        .unwrap_or_default();

    let corpus_a = remove_disambiguation_snippets(&raw_a);

    let corpus = if corpus_a.trim().is_empty() {
        let question = format!("what is {search_query}");
        let raw_b = aggregator
            .get_response(&question)
            .await
            .map(|r| corpus_from_response(&r))
            .unwrap_or_default();
        remove_disambiguation_snippets(&raw_b)
    } else {
        corpus_a
    };

    let mut kb = KnowledgeIndex::new();
    kb.ingest_text("search", &corpus);

    let chunks: Vec<String> = kb
        .query(topic, TOP_K)
        .into_iter()
        .map(|c| c.text.clone())
        .collect();

    let kb_answer = kb
        .answer(topic, TOP_K)
        .filter(|a| a.split_whitespace().count() >= MIN_KB_WORDS)
        .unwrap_or_default();

    let prose_text = if !kb_answer.is_empty() {
        let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let rule_sources: Vec<&str> = chunk_refs.iter().copied().take(12).collect();
        let rule_base = RuleBase::from_chunks(&rule_sources);
        let engine = InferenceEngine::new(INFERENCE_DEPTH, CF_THRESHOLD);

        if let Some(inference) = engine.reason(topic, &rule_base) {
            let expert_prose = ProseBuilder::new(2, 5).build(&inference, &chunk_refs, seed);
            if expert_prose.split_whitespace().count() >= 30 {
                let deduplicator = GlobalDeduplicator::new(0.65);
                let merged = format!("{}\n\n{}", expert_prose, kb_answer);
                deduplicator.deduplicate_text(&merged)
            } else {
                kb_answer
            }
        } else {
            kb_answer
        }
    } else {
        corpus.clone()
    };

    if !prose_text.trim().is_empty() {
        let assembled = assemble(&prose_text, topic, seed);
        if !assembled.trim().is_empty() {
            let enhancer = StochasticEnhancer::new(STOCHASTIC_PROB);
            return enhancer.enhance(&assembled);
        }
    }

    if !corpus.is_empty() {
        if let Ok(sents) = TextSummarizer::new(5, 60, 5).summarize_with_query(&corpus, topic) {
            if !sents.is_empty() {
                let enhancer = StochasticEnhancer::new(STOCHASTIC_PROB);
                return enhancer.enhance(&sents.join(" "));
            }
        }
        return corpus;
    }

    let fallback_phrases = [
        "is a foundational topic with many interconnected aspects.",
        "spans many dimensions of knowledge and ongoing research.",
        "continues to be studied and discussed across disciplines.",
        "has shaped thought and practice in significant ways.",
        "is widely explored across academic and professional fields.",
    ];
    format!(
        "{} {}",
        topic,
        fallback_phrases[(turn as usize) % fallback_phrases.len()]
    )
}

const FALSE_TOPIC_WORDS: &[&str] = &[
    "given",
    "axiom",
    "theorem",
    "lemma",
    "corollary",
    "proof",
    "definition",
    "statement",
    "conditional",
    "probability",
    "since",
    "however",
    "therefore",
    "thus",
    "hence",
    "whereas",
    "although",
    "despite",
    "section",
    "chapter",
    "figure",
    "note",
    "remark",
    "example",
    "case",
    "result",
    "summary",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "various",
    "several",
    "other",
    "another",
    "certain",
    "general",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "also",
    "often",
    "usually",
    "commonly",
    "american",
    "european",
    "asian",
    "african",
    "arabic",
    "arab",
    "french",
    "british",
    "german",
    "japanese",
    "chinese",
    "russian",
    "italian",
    "spanish",
    "portuguese",
    "dutch",
    "swedish",
    "norwegian",
    "danish",
    "polish",
    "turkish",
    "persian",
    "greek",
    "hebrew",
    "byzantine",
    "ottoman",
    "roman",
    "phoenician",
    "lebanese",
    "syrian",
    "iraqi",
    "iranian",
    "egyptian",
    "moroccan",
    "algerian",
    "tunisian",
    "australian",
    "canadian",
    "brazilian",
    "mexican",
    "indian",
    "korean",
    "vietnamese",
    "thai",
    "indonesian",
    "pakistani",
    "bangladeshi",
    "nigerian",
    "kenyan",
    "ghanaian",
    "south",
    "north",
    "east",
    "west",
    "eastern",
    "western",
    "northern",
    "southern",
    "central",
    "middle",
    "modern",
    "ancient",
    "medieval",
    "classical",
    "contemporary",
    "traditional",
    "early",
    "late",
    "new",
    "old",
    "graydon",
    "brendan",
    "guido",
    "james",
    "anders",
    "grace",
    "alan",
    "linus",
    "dennis",
    "ken",
    "brian",
    "bjarne",
    "larry",
    "john",
    "simon",
    "xavier",
    "stroustrup",
    "ritchie",
    "thompson",
    "eich",
    "rossum",
    "hoare",
    "wirth",
    "mccarthy",
    "andrei",
    "alexandrescu",
    "zach",
    "condon",
    "donald",
    "knuth",
    "turing",
    "dijkstra",
    "mozilla",
    "google",
    "microsoft",
    "apple",
    "amazon",
    "facebook",
    "meta",
    "oracle",
    "ibm",
    "linux",
    "unix",
    "windows",
    "macos",
    "java",
    "ocaml",
    "haskell",
    "erlang",
    "kotlin",
    "swift",
    "scala",
    "clojure",
    "lisp",
    "prolog",
    "fortran",
    "cobol",
    "pascal",
    "delphi",
    "typescript",
    "javascript",
    "coffeescript",
    "ruby",
    "perl",
    "php",
    "elixir",
    "dart",
    "julia",
    "matlab",
    "assembly",
    "forth",
    "scheme",
    "racket",
    "smalltalk",
    "groovy",
    "crystal",
    "nim",
    "zig",
    "fsharp",
    "csharp",
    "objective",
    "powershell",
    "bash",
    "shell",
    "python",
    "syntax",
    "semantics",
    "abstraction",
    "paradigm",
    "concept",
    "approach",
    "technique",
    "mechanism",
    "feature",
    "aspect",
    "wikipedia",
    "wikidata",
    "design",
    "patterns",
    "major",
    "largest",
    "smallest",
    "greatest",
    "highest",
    "lowest",
    "primary",
    "secondary",
    "significant",
    "important",
    "notable",
    "special",
    "unique",
    "various",
    "multiple",
    "numerous",
];

const STOPWORDS: &[&str] = &[
    "the", "this", "that", "these", "those", "a", "an", "and", "but", "or", "nor", "yet", "so",
    "for", "both", "of", "in", "on", "at", "to", "from", "with", "by", "as", "into", "over",
    "under", "about", "above", "below", "between", "through", "before", "after", "since", "until",
    "while", "within", "without", "it", "its", "they", "their", "them", "we", "our", "you", "your",
    "who", "which", "what", "when", "where", "how", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "do", "does", "did", "will", "would", "can", "could", "may",
    "might", "shall", "should", "must", "make", "made", "get", "gets", "got", "use", "uses",
    "used", "using", "not", "only", "just", "even", "also", "back", "well", "very", "first",
    "second", "third", "last", "next", "include", "includes", "however", "than", "then",
];

const GENERIC_NOUNS: &[&str] = &[
    "computing",
    "computer",
    "programming",
    "language",
    "languages",
    "software",
    "hardware",
    "system",
    "systems",
    "process",
    "processing",
    "data",
    "information",
    "technology",
    "implementation",
    "interface",
    "library",
    "framework",
    "runtime",
    "compiler",
    "interpreter",
    "modern",
    "advanced",
    "basic",
    "simple",
    "complex",
    "world",
    "country",
    "region",
    "area",
    "place",
    "city",
    "town",
    "people",
    "person",
    "group",
    "organization",
    "community",
    "society",
    "history",
    "culture",
    "time",
    "period",
    "century",
    "decade",
    "years",
    "research",
    "study",
    "field",
    "domain",
    "theory",
    "practice",
];

const PERSON_TITLES: &[&str] = &[
    "developer",
    "architect",
    "engineer",
    "researcher",
    "professor",
    "scientist",
    "creator",
    "inventor",
    "designer",
    "coder",
    "programmer",
    "author",
    "founder",
    "ceo",
    "cto",
    "phd",
    "student",
    "graduate",
    "undergraduate",
    "lecturer",
    "scholar",
    "musician",
    "singer",
    "songwriter",
    "artist",
    "painter",
    "composer",
    "writer",
    "journalist",
    "director",
    "producer",
    "actor",
    "actress",
    "politician",
    "president",
    "minister",
    "senator",
    "general",
];

fn is_person_name_context(words: &[&str], i: usize) -> bool {
    for j in i.saturating_sub(3)..i {
        let w = words[j].to_lowercase();
        let w_clean = w.trim_matches(|c: char| !c.is_alphabetic());
        if PERSON_TITLES.contains(&w_clean) {
            return true;
        }
    }
    if i > 0 {
        let prev = words[i - 1].to_lowercase();
        let prev_clean = prev.trim_matches(|c: char| !c.is_alphabetic());
        if prev_clean == "by" {
            if let Some(&next) = words.get(i + 1) {
                let next_core: String = next.chars().filter(|c| c.is_alphabetic()).collect();
                if next_core.len() >= 2 {
                    let mut nc = next_core.chars();
                    if nc.next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn is_bad_response(text: &str) -> bool {
    if text.contains("spans many interconnected areas")
        || text.contains("spans many dimensions of knowledge")
        || text.contains("active area of research and practical application")
        || text.contains("continues to evolve with contributions")
        || text.contains("underpins key ideas in systems")
        || text.contains("widely studied across academia")
        || text.contains("foundational topic in modern software")
        || text.contains("spans many dimensions")
        || text.contains("is a foundational topic with many")
        || text.contains("spans many dimensions of knowledge")
        || text.contains("has shaped thought and practice")
        || text.contains("is widely explored across academic")
        || text.contains("continues to be studied and discussed")
    {
        return true;
    }
    if text.matches("...").count() >= 2 {
        return true;
    }
    if text.split_whitespace().count() < 20 {
        return true;
    }
    false
}

const TECH_COMPOUNDS: &[&str] = &[
    "ownership model",
    "borrow checker",
    "type system",
    "type inference",
    "pattern matching",
    "algebraic data types",
    "higher-order functions",
    "memory safety",
    "memory management",
    "garbage collection",
    "concurrency model",
    "async runtime",
    "async programming",
    "error handling",
    "trait objects",
    "generic programming",
    "zero-cost abstractions",
    "unsafe code",
    "foreign function interface",
    "package manager",
    "build system",
    "standard library",
    "functional programming",
    "object-oriented programming",
    "systems programming",
    "embedded systems",
    "WebAssembly",
    "LLVM backend",
    "compiler design",
];

pub fn summarize_to_topic(text: &str, domain: &str, current_topic: &str, turn: u64) -> String {
    let clean = text.trim();
    if clean.is_empty() {
        return aspect_topic(domain, turn + 1);
    }

    if is_bad_response(clean) {
        return aspect_topic(domain, turn + 1);
    }

    let first_sentences: Vec<&str> = clean
        .split(|c| c == '.' || c == '!' || c == '?')
        .take(3)
        .collect();

    for sentence in &first_sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        for (i, &word) in words.iter().enumerate() {
            if i == 0 || word.len() < 4 {
                continue;
            }
            if word.contains('-') || word.contains('_') {
                continue;
            }
            let core: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            if core.len() < 4 {
                continue;
            }
            let mut chars = core.chars();
            let first_char = match chars.next() {
                Some(c) => c,
                None => continue,
            };
            if !first_char.is_uppercase() {
                continue;
            }
            let core_lower = core.to_lowercase();
            if STOPWORDS.contains(&core_lower.as_str()) {
                continue;
            }
            if FALSE_TOPIC_WORDS.contains(&core_lower.as_str()) {
                continue;
            }
            if is_person_name_context(&words, i) {
                continue;
            }

            let rest_all_lower = core.chars().skip(1).all(|c| c.is_lowercase());

            if rest_all_lower {
                let domain_words: Vec<&str> = domain.split_whitespace().collect();
                let pivot_in_domain = domain_words
                    .iter()
                    .any(|&dw| dw.to_lowercase() == core_lower);

                if pivot_in_domain {
                    if let Some(pivot_pos) = domain_words
                        .iter()
                        .position(|&dw| dw.to_lowercase() == core_lower)
                    {
                        if let Some(&next_dw) = domain_words.get(pivot_pos + 1) {
                            if next_dw.len() >= 4
                                && !STOPWORDS.contains(&next_dw.to_lowercase().as_str())
                            {
                                let candidate = format!("{} {}", core, next_dw);
                                if !candidate.eq_ignore_ascii_case(current_topic) {
                                    return candidate;
                                }
                            }
                        }
                    }
                } else {
                    if !core.eq_ignore_ascii_case(current_topic) {
                        return core;
                    }
                }
            } else {
                if !core.eq_ignore_ascii_case(current_topic) {
                    return core;
                }
            }
        }
    }

    if is_tech_domain(domain) {
        let clean_lower = clean.to_lowercase();
        let key = domain_key(domain);
        for &compound in TECH_COMPOUNDS {
            if clean_lower.contains(&compound.to_lowercase()) {
                let candidate = format!("{} {}", key, compound);
                if !candidate.eq_ignore_ascii_case(current_topic)
                    && !compound.eq_ignore_ascii_case(current_topic)
                {
                    return candidate;
                }
            }
        }
    }

    for word in clean.split_whitespace() {
        if word.len() < 5 || word.contains('-') {
            continue;
        }
        let core: String = word.chars().filter(|c| c.is_alphabetic()).collect();
        if core.len() < 5 {
            continue;
        }
        let mut ch = core.chars();
        let up = ch.next().map(|c| c.is_uppercase()).unwrap_or(false);
        let lo_rest = ch.all(|c| c.is_lowercase());
        if !up || !lo_rest {
            continue;
        }
        let lo = core.to_lowercase();
        if STOPWORDS.contains(&lo.as_str()) || FALSE_TOPIC_WORDS.contains(&lo.as_str()) {
            continue;
        }
        if lo == domain_key(domain).to_lowercase() {
            continue;
        }
        if !core.eq_ignore_ascii_case(current_topic) {
            return core;
        }
    }

    let first_sentence = clean
        .split(|c| c == '.' || c == '!' || c == '?')
        .next()
        .unwrap_or(clean);

    let content_words: Vec<&str> = first_sentence
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| {
            let lo = w.to_lowercase();
            w.len() >= 7
                && !STOPWORDS.contains(&lo.as_str())
                && !FALSE_TOPIC_WORDS.contains(&lo.as_str())
                && !GENERIC_NOUNS.contains(&lo.as_str())
                && !w.contains('-')
                && w.chars().any(|c| c.is_alphabetic())
        })
        .take(2)
        .collect();

    if content_words.len() >= 2 {
        let candidate = content_words.join(" ");
        if !candidate.eq_ignore_ascii_case(current_topic) {
            return candidate;
        }
    }

    aspect_topic(domain, turn + 1)
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
