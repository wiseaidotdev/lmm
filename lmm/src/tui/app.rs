// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.

use crate::tui::{
    state::{AppTab, TuiEvent, TuiState},
    tabs::{
        analytics::render_analytics_tab,
        main::{render_main_tab, render_status_bar, render_tab_bar},
        settings::render_settings_tab,
        theme::render_theme_tab,
    },
};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{io, path::Path, time::Duration};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tui_input::backend::crossterm::EventHandler;

pub struct TuiApp {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    state: TuiState,
    abort_token: Arc<AtomicBool>,
    event_tx: UnboundedSender<TuiEvent>,
}

impl TuiApp {
    pub fn new(
        receiver: UnboundedReceiver<TuiEvent>,
        event_tx: UnboundedSender<TuiEvent>,
        abort_token: Arc<AtomicBool>,
    ) -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        let state = TuiState::new(receiver);

        Ok(Self {
            terminal,
            state,
            abort_token,
            event_tx,
        })
    }

    pub fn run(mut self) -> Result<()> {
        loop {
            self.state.tick();
            self.state.process_events();

            let palette = self.state.theme_config.palette();
            let show_cursor = (self.state.active_tab == AppTab::Main
                && self.state.pipeline_status == "Idle")
                || (self.state.active_tab == AppTab::Settings && self.state.settings_editing);

            self.terminal.draw(|frame| {
                let area = frame.area();

                let vertical = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(2),
                        Constraint::Fill(1),
                        Constraint::Length(2),
                    ])
                    .split(area);

                render_tab_bar(frame, vertical[0], &self.state, &palette);

                match self.state.active_tab {
                    AppTab::Main => render_main_tab(frame, vertical[1], &self.state, &palette),
                    AppTab::Analytics => {
                        render_analytics_tab(frame, vertical[1], &self.state, &palette)
                    }
                    AppTab::Theme => render_theme_tab(frame, vertical[1], &self.state, &palette),
                    AppTab::Settings => {
                        render_settings_tab(frame, vertical[1], &self.state, &palette)
                    }
                }

                render_status_bar(frame, vertical[2], &self.state, &palette);
            })?;

            if show_cursor {
                self.terminal.show_cursor()?;
            } else {
                self.terminal.hide_cursor()?;
            }

            if self.state.should_quit {
                self.state.tram_settings.save();
                break;
            }

            if event::poll(Duration::from_millis(50))?
                && let Ok(Event::Key(key)) = event::read()
                && key.kind == KeyEventKind::Press
            {
                self.handle_key(key.code, key.modifiers)?;
            }
        }

        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;
        Ok(())
    }

    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Result<()> {
        match code {
            KeyCode::Char('q') if modifiers.is_empty() => {
                self.state.should_quit = true;
            }
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.state.should_quit = true;
            }
            KeyCode::Tab => {
                self.state.active_tab = self.state.active_tab.next();
            }
            KeyCode::BackTab => {
                self.state.active_tab = self.state.active_tab.prev();
            }
            KeyCode::Char('1') if modifiers.contains(KeyModifiers::ALT) => {
                self.state.active_tab = AppTab::Main
            }
            KeyCode::Char('2') if modifiers.contains(KeyModifiers::ALT) => {
                self.state.active_tab = AppTab::Analytics
            }
            KeyCode::Char('3') if modifiers.contains(KeyModifiers::ALT) => {
                self.state.active_tab = AppTab::Theme
            }
            KeyCode::Char('4') if modifiers.contains(KeyModifiers::ALT) => {
                self.state.active_tab = AppTab::Settings
            }
            _ => match self.state.active_tab {
                AppTab::Main => self.handle_main_key(code, modifiers),
                AppTab::Theme => self.handle_theme_key(code),
                AppTab::Settings => self.handle_settings_key(code, modifiers),
                AppTab::Analytics => {}
            },
        }
        Ok(())
    }

    fn handle_main_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        match code {
            KeyCode::Enter => {
                let text = self.state.input_buffer.value().trim().to_string();
                if !text.is_empty() {
                    self.state.push_log(format!("❯ {}", text));
                    self.state.pipeline_status = "Searching...".to_string();
                    self.state.input_buffer = tui_input::Input::default();
                    self.state.log_scroll_offset = self.state.log_lines.len().saturating_sub(5);
                    self.state.log_h_scroll_offset = 0;

                    let tx = self.event_tx.clone();
                    let settings = self.state.tram_settings.clone();
                    let pipeline = self.state.tram_pipeline.clone().unwrap();
                    let _abort = self.abort_token.clone();

                    tokio::spawn(async move {
                        let _ = tx.send(TuiEvent::Log("Running TRAM pipeline...".to_string()));

                        let mut index = crate::cognition::knowledge::KnowledgeIndex::new();

                        #[cfg(feature = "net")]
                        if let Some(kb_path) = settings.knowledge_base {
                            let path = Path::new(&kb_path);
                            if path.exists() {
                                let _ = tx.send(TuiEvent::Log(format!(
                                    "Loading knowledge base from: {}",
                                    kb_path
                                )));
                                let source = if path.is_file() {
                                    Some(crate::cognition::knowledge::KnowledgeSource::File(
                                        path.to_path_buf(),
                                    ))
                                } else if path.is_dir() {
                                    Some(crate::cognition::knowledge::KnowledgeSource::Dir(
                                        path.to_path_buf(),
                                    ))
                                } else {
                                    None
                                };
                                if let Some(src) = source {
                                    match crate::cognition::knowledge::ingest(&mut index, src).await
                                    {
                                        Ok(n) => {
                                            let _ = tx.send(TuiEvent::Log(format!(
                                                "✓ KB ingested: {} chunks",
                                                n
                                            )));
                                        }
                                        Err(e) => {
                                            let _ = tx.send(TuiEvent::Log(format!(
                                                "✗ KB ingestion failed: {}",
                                                e
                                            )));
                                        }
                                    }
                                }
                            }
                        }

                        let _ = tx.send(TuiEvent::PipelineStatus("Decomposing...".to_string()));

                        let result = match pipeline.run(&text, &mut index).await {
                            Ok(r) => r,
                            Err(e) => {
                                let _ = tx.send(TuiEvent::Log(format!("✗ Pipeline error: {}", e)));
                                let _ = tx.send(TuiEvent::PipelineStatus("Idle".to_string()));
                                return;
                            }
                        };

                        let _ = tx.send(TuiEvent::PipelineStatus("Thinking...".to_string()));

                        if let Some(ate) = &result.ate_result {
                            let _ = tx.send(TuiEvent::Log(format!(
                                "🧠 ATE: mode={} steps={} confidence={:.3}",
                                ate.mode.label(),
                                ate.steps,
                                ate.confidence
                            )));
                        }

                        let _ = tx.send(TuiEvent::PipelineStatus("Reasoning...".to_string()));
                        let _ = tx.send(TuiEvent::Log(format!(
                            "🔄 ThinkLoop: converged={} steps={} error={:.3}",
                            result.thinkloop_result.converged,
                            result.thinkloop_result.steps,
                            result.thinkloop_result.final_error
                        )));

                        let _ = tx.send(TuiEvent::PipelineStatus("Retrieving...".to_string()));
                        let _ = tx.send(TuiEvent::Log(format!(
                            "📚 KB retrieval: {} chunks used",
                            result.kb_retrieval_result.kb_chunks.len()
                        )));

                        let _ = tx.send(TuiEvent::PipelineStatus("Inferring...".to_string()));
                        let _ = tx.send(TuiEvent::Log(format!(
                            "🎯 Expert inference: depth={} cf_threshold={:.2}",
                            settings.inference_depth, settings.cf_threshold
                        )));

                        let _ = tx.send(TuiEvent::PipelineStatus("Enhancing...".to_string()));
                        let _ = tx.send(TuiEvent::Log(format!(
                            "🎲 Stochastic enhancement: {} substitutions",
                            result.stochastic_result.substitutions
                        )));

                        let _ = tx.send(TuiEvent::PipelineStatus("Framing...".to_string()));

                        let _ = tx.send(TuiEvent::PipelineStatus("Composing...".to_string()));

                        let response = result.final_text;
                        let _ = tx.send(TuiEvent::Log("✓ Response generated".to_string()));
                        let _ = tx.send(TuiEvent::Response(response.clone()));

                        let web_chunks = result.web_fetch_result.total_chunks_ingested;
                        let kb_chunks = result.kb_retrieval_result.kb_chunks.len();

                        let _ = tx.send(TuiEvent::IncRequest);
                        let _ = tx.send(TuiEvent::IncResponse);
                        let _ = tx.send(TuiEvent::AddWebChunks(web_chunks));
                        let _ = tx.send(TuiEvent::AddKbChunks(kb_chunks));
                        let _ = tx.send(TuiEvent::PipelineStatus("Done".to_string()));

                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                        let _ = tx.send(TuiEvent::PipelineStatus("Idle".to_string()));
                    });
                }
            }
            KeyCode::Esc => {
                self.abort_token.store(true, Ordering::SeqCst);
            }
            KeyCode::PageUp => {
                let height: usize = 20;
                self.state.log_scroll_offset = self
                    .state
                    .log_scroll_offset
                    .saturating_sub(height.saturating_sub(2));
            }
            KeyCode::PageDown => {
                let height: usize = 20;
                self.state.log_scroll_offset = self
                    .state
                    .log_scroll_offset
                    .saturating_add(height.saturating_sub(2));
            }
            KeyCode::Up if modifiers.is_empty() => {
                self.state.log_scroll_offset = self.state.log_scroll_offset.saturating_sub(1);
            }
            KeyCode::Down if modifiers.is_empty() => {
                self.state.log_scroll_offset = self.state.log_scroll_offset.saturating_add(1);
            }
            KeyCode::Left if modifiers.is_empty() => {
                self.state.log_h_scroll_offset = self.state.log_h_scroll_offset.saturating_sub(8);
            }
            KeyCode::Right if modifiers.is_empty() => {
                self.state.log_h_scroll_offset = self.state.log_h_scroll_offset.saturating_add(8);
            }
            other => {
                let ev = Event::Key(KeyEvent::new(other, modifiers));
                self.state.input_buffer.handle_event(&ev);
            }
        }
    }

    fn handle_theme_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up if self.state.selected_theme_idx > 0 => {
                self.state.selected_theme_idx -= 1;
            }
            KeyCode::Down
                if self.state.selected_theme_idx + 1
                    < crate::tui::theme::ALL_THEME_VARIANTS.len() =>
            {
                self.state.selected_theme_idx += 1;
            }
            KeyCode::Enter => {
                let variant = crate::tui::theme::ALL_THEME_VARIANTS[self.state.selected_theme_idx];
                self.state.apply_theme_variant(variant);
            }
            KeyCode::Char('r') => {
                self.state
                    .apply_theme_variant(crate::tui::theme::ThemeVariant::Dark);
                self.state.selected_theme_idx = 0;
            }
            _ => {}
        }
    }

    fn handle_settings_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        let total_fields = 28;
        let visible_fields = 8;

        if self.state.settings_editing {
            match code {
                KeyCode::Enter => {
                    let idx = self.state.settings_focus_idx;
                    let value = self.state.settings_edit_input.value().trim().to_string();

                    let success = self.apply_setting_value(idx, &value);
                    if success {
                        self.state.settings_editing = false;
                        self.state.settings_edit_input = tui_input::Input::default();
                        self.state.push_log("✓ Setting updated".to_string());
                    } else {
                        self.state.push_log("✗ Invalid value".to_string());
                    }
                }
                KeyCode::Esc => {
                    self.state.settings_editing = false;
                    self.state.settings_edit_input = tui_input::Input::default();
                }
                _other => {
                    let ev = Event::Key(KeyEvent::new(code, modifiers));
                    self.state.settings_edit_input.handle_event(&ev);
                }
            }
            return;
        }

        match code {
            KeyCode::Up => {
                self.state.settings_focus_idx = self.state.settings_focus_idx.saturating_sub(1);
                if self.state.settings_focus_idx < self.state.settings_scroll_offset {
                    self.state.settings_scroll_offset = self.state.settings_focus_idx;
                }
            }
            KeyCode::Down => {
                if self.state.settings_focus_idx + 1 < total_fields {
                    self.state.settings_focus_idx += 1;
                    if self.state.settings_focus_idx
                        >= self.state.settings_scroll_offset + visible_fields
                    {
                        self.state.settings_scroll_offset = self
                            .state
                            .settings_focus_idx
                            .saturating_sub(visible_fields - 1);
                    }
                }
            }
            KeyCode::Char('e') if modifiers.is_empty() => {
                if self.state.settings_focus_idx >= 4 {
                    self.state.settings_editing = true;
                    let current_value = self.get_setting_value(self.state.settings_focus_idx);
                    self.state.settings_edit_input =
                        tui_input::Input::default().with_value(current_value);
                }
            }
            KeyCode::Char('s') if modifiers.is_empty() => {
                self.state.save_settings();
                self.state.reinitialize_pipeline();
                self.state
                    .push_log("Settings saved and pipeline reinitialized".to_string());
            }
            KeyCode::Char('s') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.state.save_settings();
                self.state.reinitialize_pipeline();
                self.state
                    .push_log("Settings saved and pipeline reinitialized".to_string());
            }
            KeyCode::Char(' ') | KeyCode::Enter => match self.state.settings_focus_idx {
                0 => self.state.tram_settings.deep = !self.state.tram_settings.deep,
                1 => self.state.tram_settings.think = !self.state.tram_settings.think,
                2 => self.state.tram_settings.no_web = !self.state.tram_settings.no_web,
                3 => self.state.tram_settings.no_frame = !self.state.tram_settings.no_frame,
                _ => {}
            },
            _ => {}
        }
    }

    fn get_setting_value(&self, idx: usize) -> String {
        match idx {
            4 => self
                .state
                .tram_settings
                .knowledge_base
                .clone()
                .unwrap_or_default(),
            5 => self.state.tram_settings.iterations.to_string(),
            6 => self.state.tram_settings.inference_depth.to_string(),
            7 => self.state.tram_settings.cf_threshold.to_string(),
            8 => self.state.tram_settings.think_steps.to_string(),
            9 => self.state.tram_settings.think_threshold.to_string(),
            10 => self.state.tram_settings.think_patience.to_string(),
            11 => self.state.tram_settings.max_tasks.to_string(),
            12 => self.state.tram_settings.probability.to_string(),
            13 => self.state.tram_settings.convergence_threshold.to_string(),
            14 => self.state.tram_settings.k_proportional.to_string(),
            15 => self.state.tram_settings.k_integral.to_string(),
            16 => self.state.tram_settings.top_k_min.to_string(),
            17 => self.state.tram_settings.top_k_var.to_string(),
            18 => self.state.tram_settings.min_rule_count.to_string(),
            19 => self.state.tram_settings.min_expert_words.to_string(),
            20 => self.state.tram_settings.dedup_threshold.to_string(),
            21 => self.state.tram_settings.prose_min_sentences.to_string(),
            22 => self.state.tram_settings.prose_max_sentences.to_string(),
            23 => self.state.tram_settings.min_kb_words.to_string(),
            24 => self.state.tram_settings.min_prose_words.to_string(),
            25 => self.state.tram_settings.rule_source_limit.to_string(),
            26 => self.state.tram_settings.use_assembly.to_string(),
            27 => self.state.tram_settings.use_summarizer_fallback.to_string(),
            _ => String::new(),
        }
    }

    fn apply_setting_value(&mut self, idx: usize, value: &str) -> bool {
        match idx {
            4 => {
                self.state.tram_settings.knowledge_base = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                };
                true
            }
            5 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.iterations = v;
                    true
                } else {
                    false
                }
            }
            6 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.inference_depth = v;
                    true
                } else {
                    false
                }
            }
            7 => {
                if let Ok(v) = value.parse::<f64>() {
                    self.state.tram_settings.cf_threshold = v;
                    true
                } else {
                    false
                }
            }
            8 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.think_steps = v;
                    true
                } else {
                    false
                }
            }
            9 => {
                if let Ok(v) = value.parse::<f64>() {
                    self.state.tram_settings.think_threshold = v;
                    true
                } else {
                    false
                }
            }
            10 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.think_patience = v;
                    true
                } else {
                    false
                }
            }
            11 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.max_tasks = v;
                    true
                } else {
                    false
                }
            }
            12 => {
                if let Ok(v) = value.parse::<f64>() {
                    self.state.tram_settings.probability = v;
                    true
                } else {
                    false
                }
            }
            13 => {
                if let Ok(v) = value.parse::<f64>() {
                    self.state.tram_settings.convergence_threshold = v;
                    true
                } else {
                    false
                }
            }
            14 => {
                if let Ok(v) = value.parse::<f64>() {
                    self.state.tram_settings.k_proportional = v;
                    true
                } else {
                    false
                }
            }
            15 => {
                if let Ok(v) = value.parse::<f64>() {
                    self.state.tram_settings.k_integral = v;
                    true
                } else {
                    false
                }
            }
            16 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.top_k_min = v;
                    true
                } else {
                    false
                }
            }
            17 => {
                if let Ok(v) = value.parse::<u64>() {
                    self.state.tram_settings.top_k_var = v;
                    true
                } else {
                    false
                }
            }
            18 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.min_rule_count = v;
                    true
                } else {
                    false
                }
            }
            19 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.min_expert_words = v;
                    true
                } else {
                    false
                }
            }
            20 => {
                if let Ok(v) = value.parse::<f64>() {
                    self.state.tram_settings.dedup_threshold = v;
                    true
                } else {
                    false
                }
            }
            21 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.prose_min_sentences = v;
                    true
                } else {
                    false
                }
            }
            22 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.prose_max_sentences = v;
                    true
                } else {
                    false
                }
            }
            23 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.min_kb_words = v;
                    true
                } else {
                    false
                }
            }
            24 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.min_prose_words = v;
                    true
                } else {
                    false
                }
            }
            25 => {
                if let Ok(v) = value.parse::<usize>() {
                    self.state.tram_settings.rule_source_limit = v;
                    true
                } else {
                    false
                }
            }
            26 => {
                if let Ok(v) = value.parse::<bool>() {
                    self.state.tram_settings.use_assembly = v;
                    true
                } else {
                    false
                }
            }
            27 => {
                if let Ok(v) = value.parse::<bool>() {
                    self.state.tram_settings.use_summarizer_fallback = v;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

pub async fn run_tui() -> Result<()> {
    use tokio::sync::mpsc::unbounded_channel;

    let (tx, rx) = unbounded_channel::<TuiEvent>();
    let abort_token = Arc::new(AtomicBool::new(false));

    let tx_clone = tx.clone();
    tokio::spawn(async move {
        tx_clone
            .send(TuiEvent::Log(
                "Enter a prompt to start the LMM pipeline.".to_string(),
            ))
            .ok();
    });

    let app = TuiApp::new(rx, tx, abort_token.clone())?;
    app.run()?;

    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.
