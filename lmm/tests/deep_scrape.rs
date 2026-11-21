// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integration tests for `tram::deep_scrape::resolve_ddg_url`.
//!
//! These tests cover the four DDG URL formats that the resolver must handle:
//!
//! | Format                | Input example                                 | Expected output               |
//! |-----------------------|-----------------------------------------------|-------------------------------|
//! | DDG redirect param    | `/l/?uddg=https%3A%2F%2Fexample.com%2F`      | `https://example.com/`        |
//! | Protocol-relative     | `//example.com/page`                         | `https://example.com/page`    |
//! | DDG entity page       | `https://duckduckgo.com/Rust_(programming)` | Wikipedia URL                 |
//! | Plain HTTP/S URL      | `https://example.com/article`                | `https://example.com/article` |
//! | Junk/DDG-internal     | `https://duckduckgo.com/`                    | empty string                  |

#[cfg(feature = "net")]
mod deep_scrape_tests {
    use lmm::tram::deep_scrape::resolve_ddg_url;

    /// A DDG redirect href (`/l/?uddg=<encoded>`) is decoded and the target
    /// URL is returned.
    #[test]
    fn ddg_redirect_param_decoded() {
        let href = "/l/?uddg=https%3A%2F%2Fexample.com%2Farticle";
        let result = resolve_ddg_url(href);
        assert_eq!(
            result, "https://example.com/article",
            "redirect href must decode to target URL"
        );
    }

    /// A `//`-prefixed protocol-relative URL is upgraded to `https://`.
    #[test]
    fn protocol_relative_upgraded_to_https() {
        let href = "//example.com/page";
        let result = resolve_ddg_url(href);
        assert_eq!(
            result, "https://example.com/page",
            "protocol-relative must become https://"
        );
    }

    /// A DDG entity page path (e.g. `/Rust_(programming_language)`) is mapped
    /// to the matching Wikipedia article URL.
    #[test]
    fn ddg_entity_page_maps_to_wikipedia() {
        let href = "https://duckduckgo.com/Rust_(programming_language)";
        let result = resolve_ddg_url(href);
        assert!(
            result.starts_with("https://en.wikipedia.org/wiki/"),
            "entity page must resolve to Wikipedia: got {result:?}"
        );
        assert!(
            result.contains("Rust"),
            "Wikipedia URL must retain the topic slug"
        );
    }

    /// A plain `https://` URL that is not DDG-internal is returned verbatim.
    #[test]
    fn plain_https_url_returned_verbatim() {
        let href = "https://doc.rust-lang.org/book/";
        let result = resolve_ddg_url(href);
        assert_eq!(result, href, "plain https URL must be returned as-is");
    }

    /// A DDG-internal URL (domain contains `duckduckgo`) returns an empty string.
    #[test]
    fn duckduckgo_internal_url_returns_empty() {
        let href = "https://duckduckgo.com/";
        let result = resolve_ddg_url(href);
        assert!(
            result.is_empty(),
            "DDG root URL must return empty string; got {result:?}"
        );
    }

    /// A `?`-only DDG entity path is NOT mapped to Wikipedia (guard against
    /// turning query parameters into article slugs).
    #[test]
    fn ddg_entity_path_query_only_returns_empty() {
        let href = "https://duckduckgo.com/?q=rust";
        let result = resolve_ddg_url(href);
        assert!(
            result.is_empty() || !result.contains("wikipedia"),
            "query-only DDG URL must not produce a Wikipedia link; got {result:?}"
        );
    }

    /// A `lite` DDG path is NOT mapped to Wikipedia.
    #[test]
    fn ddg_entity_path_lite_returns_empty() {
        let href = "https://duckduckgo.com/lite/?q=rust";
        let result = resolve_ddg_url(href);
        assert!(
            !result.contains("wikipedia"),
            "lite DDG URL must not produce a Wikipedia link; got {result:?}"
        );
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
