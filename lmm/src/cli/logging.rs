// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # CLI Logging Setup
//!
//! Configures tracing with custom formatters for console and file output.

use anyhow::Result;
use std::fmt;
use std::io;
use tracing::Subscriber;
use tracing_appender::rolling;
use tracing_subscriber::{
    Layer, Registry, filter,
    fmt::{self as ts_fmt, FmtContext, FormatEvent, FormatFields},
    layer::SubscriberExt,
    registry::LookupSpan,
};

#[cfg(feature = "cli")]
struct NoLevelFormatter;

#[cfg(feature = "cli")]
impl<S, N> FormatEvent<S, N> for NoLevelFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: ts_fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> fmt::Result {
        ctx.format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

/// Sets up the tracing subscriber with console and file layers.
///
/// # Arguments
///
/// * `verbose` - When true, INFO level to console; otherwise WARN.
///
/// # Time Complexity
///
/// O(1)
#[cfg(feature = "cli")]
pub fn setup_logging(verbose: bool) -> Result<()> {
    let console_level = if verbose {
        filter::LevelFilter::INFO
    } else {
        filter::LevelFilter::WARN
    };
    let file_appender = rolling::daily("logs", "lmm_log");

    let console_layer = ts_fmt::Layer::new()
        .compact()
        .without_time()
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_target(false)
        .with_writer(io::stdout)
        .event_format(NoLevelFormatter)
        .with_filter(console_level);

    let file_layer = ts_fmt::Layer::new()
        .compact()
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_target(true)
        .with_writer(file_appender)
        .with_filter(filter::LevelFilter::DEBUG);

    let subscriber = Registry::default().with(console_layer).with(file_layer);
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
