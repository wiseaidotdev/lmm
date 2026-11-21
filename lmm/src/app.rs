// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Application Entry-Point
//!
//! This module wires the `crate::cli::commands` argument parser to the
//! individual subsystem handler modules.  Each command variant delegates to a
//! dedicated handler so this file stays a thin dispatcher with no business
//! logic of its own.
//!
//! All public functionality is gated behind the `cli` Cargo feature so the
//! library can be used as a pure Rust or Python/Node.js dependency without
//! pulling in the CLI stack.
//!
//! ## TRAM Default
//!
//! When the user invokes `lmm -p 'question'` without a subcommand, the
//! `command` field of [`crate::cli::commands::Cli`] is `None` and the
//! `prompt` field is `Some(question)`.  In that case `run_cli_entry` falls
//! through to [`tram::handle_tram`].

#[cfg(feature = "cli")]
use {
    crate::cli::{
        commands::{
            Cli,
            Commands::{
                Causal, Consciousness as CmdConsciousness, Decode, Discover, Encode, Essay,
                Field as CmdField, Imagen, Paragraph, Physics, Predict, Sentence, Simulate,
                Summarize, Tram,
            },
        },
        handlers::encode::{handle_decode, handle_encode},
        logging::setup_logging,
    },
    clap::Parser,
    std::process::exit,
};

#[cfg(all(feature = "cli", feature = "net"))]
use crate::cli::commands::Commands;

pub mod generation_commands;
pub mod helpers;
pub mod simulation_commands;
pub mod text_commands;
pub mod tram;
pub mod tram_handler;

/// Main CLI entry point.
///
/// Parses command-line arguments and dispatches to the appropriate handler.
#[cfg(feature = "cli")]
pub async fn run_cli_entry(args: Vec<String>) -> anyhow::Result<()> {
    let cli = Cli::parse_from(args);
    setup_logging(cli.verbose)?;

    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            let prompt = cli.prompt.unwrap_or_else(|| {
                tracing::error!(
                    "lmm: no subcommand given. Try `lmm --help` or `lmm -p 'your question'`."
                );
                exit(1);
            });
            return tram::handle_tram(
                prompt,
                cli.knowledge_base,
                cli.probability,
                cli.verbose,
                cli.iterations,
                cli.inference_depth,
                cli.cf_threshold,
                cli.think,
                cli.think_steps,
                cli.think_threshold,
                cli.think_patience,
                cli.max_tasks,
                cli.no_frame,
                cli.deep,
            )
            .await;
        }
    };

    match command {
        Tram {
            prompt,
            knowledge_base,
            probability,
            iterations,
            depth: _,
            inference_depth,
            cf_threshold,
            think,
            think_steps,
            think_threshold,
            think_patience,
            max_tasks,
            no_frame,
            deep,
        } => {
            tram::handle_tram(
                prompt,
                knowledge_base,
                probability,
                cli.verbose,
                iterations,
                inference_depth,
                cf_threshold,
                think,
                think_steps,
                think_threshold,
                think_patience,
                max_tasks,
                no_frame,
                deep,
            )
            .await?;
        }

        Simulate { step, steps } => {
            simulation_commands::dispatch_simulate(step, steps, cli.verbose)?;
        }

        Discover {
            data_path: _,
            iterations,
        } => {
            simulation_commands::dispatch_discover(iterations, cli.verbose)?;
        }

        CmdConsciousness { lookahead } => {
            simulation_commands::dispatch_consciousness(lookahead, cli.verbose)?;
        }

        Physics {
            model,
            steps,
            step_size,
        } => {
            simulation_commands::handle_physics(model, steps, step_size, cli.verbose)?;
        }

        Causal {
            intervene_node,
            intervene_value,
        } => {
            simulation_commands::dispatch_causal(intervene_node, intervene_value, cli.verbose)?;
        }

        CmdField { size, operation } => {
            simulation_commands::handle_field(size, operation, cli.verbose)?;
        }

        Encode {
            input,
            text,
            iterations,
            depth,
        } => {
            handle_encode(input, text, iterations, depth, cli.verbose)?;
        }

        Decode {
            equation,
            length,
            residuals,
        } => {
            handle_decode(equation, length, residuals, cli.verbose)?;
        }

        Predict {
            input,
            text,
            window,
            predict_length,
            iterations,
            depth,
            dictionary,
            stochastic,
            probability,
        } => {
            text_commands::handle_predict(text_commands::PredictConfig {
                input,
                text,
                window,
                predict_length,
                iterations,
                depth,
                dictionary,
                stochastic,
                probability,
                verbose: cli.verbose,
            })?;
        }

        Summarize {
            input,
            text,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            text_commands::handle_summarize(text_commands::SummarizeConfig {
                input,
                text,
                sentences,
                iterations,
                depth,
                stochastic,
                probability,
                verbose: cli.verbose,
            })?;
        }

        Sentence {
            input,
            text,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            text_commands::handle_sentence(
                input,
                text,
                iterations,
                depth,
                stochastic,
                probability,
                cli.verbose,
            )?;
        }

        Paragraph {
            input,
            text,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            text_commands::handle_paragraph(text_commands::ParagraphConfig {
                input,
                text,
                sentences,
                iterations,
                depth,
                stochastic,
                probability,
                verbose: cli.verbose,
            })?;
        }

        Essay {
            input,
            text,
            paragraphs,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            text_commands::handle_essay(text_commands::EssayConfig {
                input,
                text,
                paragraphs,
                sentences,
                iterations,
                depth,
                stochastic,
                probability,
                verbose: cli.verbose,
            })?;
        }

        #[cfg(feature = "net")]
        Commands::Ask {
            prompt,
            limit,
            sentences,
            region,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            generation_commands::handle_ask(generation_commands::AskConfig {
                prompt,
                limit,
                sentences,
                region,
                iterations,
                depth,
                stochastic,
                probability,
                verbose: cli.verbose,
            })
            .await?;
        }

        Imagen {
            prompt,
            width,
            height,
            components,
            style,
            palette,
            output,
        } => {
            generation_commands::handle_imagen(generation_commands::ImagenConfig {
                prompt,
                width,
                height,
                components,
                style,
                palette,
                output,
                verbose: cli.verbose,
            })?;
        }
    }

    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
