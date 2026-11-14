// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::tram::config::TramConfig;
use crate::tram::pipeline::TramPipeline;
use crate::tui::theme::{ThemePalette, ThemeVariant};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::{fs, path::PathBuf, time::Instant};
use tokio::sync::mpsc::UnboundedReceiver;

/// Events emitted by the TRAM pipeline to the TUI.
pub enum TuiEvent {
    Log(String),
    LogAppend(String),
    ClearLog,
    Thinking(String),
    Response(String),
    StatsUpdate(SessionStats),
    IncRequest,
    IncResponse,
    AddWebChunks(usize),
    AddKbChunks(usize),
    PipelineStatus(String),
    Quit,
}

/// The four top-level tabs in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppTab {
    #[default]
    Main,
    Analytics,
    Theme,
    Settings,
}

impl AppTab {
    pub fn title(self) -> &'static str {
        match self {
            Self::Main => "  Main  ",
            Self::Analytics => "  Analytics  ",
            Self::Theme => "  Theme  ",
            Self::Settings => "  Settings  ",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![Self::Main, Self::Analytics, Self::Theme, Self::Settings]
    }

    pub fn next(self) -> Self {
        match self {
            Self::Main => Self::Analytics,
            Self::Analytics => Self::Theme,
            Self::Theme => Self::Settings,
            Self::Settings => Self::Main,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Main => Self::Settings,
            Self::Analytics => Self::Main,
            Self::Theme => Self::Analytics,
            Self::Settings => Self::Theme,
        }
    }
}

/// Accumulated chart data points for the analytics tab.
#[derive(Debug, Clone, Default)]
pub struct ChartData {
    pub web_chunks_series: Vec<(f64, f64)>,
    pub kb_chunks_series: Vec<(f64, f64)>,
    pub requests_per_minute: VecDeque<u64>,
    pub responses_per_minute: VecDeque<u64>,
    pub pipeline_rate_sparkline: VecDeque<u64>,
    pub session_start: Option<Instant>,
    pub last_stats: SessionStats,
}

impl ChartData {
    pub fn record_stats(&mut self, stats: &SessionStats) {
        if self.session_start.is_none() {
            self.session_start = Some(Instant::now());
        }
        let elapsed = self.session_start.unwrap().elapsed().as_secs_f64();
        self.web_chunks_series
            .push((elapsed, stats.web_chunks_ingested as f64));
        self.kb_chunks_series
            .push((elapsed, stats.kb_chunks_used as f64));

        let delta_req = stats.requests.saturating_sub(self.last_stats.requests);
        let delta_res = stats.responses.saturating_sub(self.last_stats.responses);
        let delta_chunks = (stats.web_chunks_ingested + stats.kb_chunks_used)
            .saturating_sub(self.last_stats.web_chunks_ingested + self.last_stats.kb_chunks_used);

        if self.requests_per_minute.len() >= 30 {
            self.requests_per_minute.pop_front();
        }
        if self.responses_per_minute.len() >= 30 {
            self.responses_per_minute.pop_front();
        }
        if self.pipeline_rate_sparkline.len() >= 60 {
            self.pipeline_rate_sparkline.pop_front();
        }

        self.requests_per_minute.push_back(delta_req);
        self.responses_per_minute.push_back(delta_res);
        self.pipeline_rate_sparkline
            .push_back(delta_chunks.try_into().unwrap_or(0));
        self.last_stats = stats.clone();
    }
}

/// TRAM pipeline configuration for the TUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TramSettings {
    pub knowledge_base: Option<String>,
    pub iterations: usize,
    pub inference_depth: usize,
    pub cf_threshold: f64,
    pub think: bool,
    pub think_steps: usize,
    pub think_threshold: f64,
    pub think_patience: usize,
    pub max_tasks: usize,
    pub probability: f64,
    pub no_frame: bool,
    pub no_web: bool,
    pub deep: bool,
    pub convergence_threshold: f64,
    pub k_proportional: f64,
    pub k_integral: f64,
    pub top_k_min: usize,
    pub top_k_var: u64,
    pub min_rule_count: usize,
    pub min_expert_words: usize,
    pub dedup_threshold: f64,
    pub prose_min_sentences: usize,
    pub prose_max_sentences: usize,
    pub min_kb_words: usize,
    pub min_prose_words: usize,
    pub rule_source_limit: usize,
    pub use_assembly: bool,
    pub use_summarizer_fallback: bool,
}

impl Default for TramSettings {
    fn default() -> Self {
        Self {
            knowledge_base: None,
            iterations: 10,
            inference_depth: 5,
            cf_threshold: 0.25,
            think: false,
            think_steps: 8,
            think_threshold: 0.40,
            think_patience: 3,
            max_tasks: 4,
            probability: 0.12,
            no_frame: false,
            no_web: false,
            deep: false,
            convergence_threshold: 0.35,
            k_proportional: 1.2,
            k_integral: 0.08,
            top_k_min: 6,
            top_k_var: 5,
            min_rule_count: 3,
            min_expert_words: 25,
            dedup_threshold: 0.70,
            prose_min_sentences: 2,
            prose_max_sentences: 7,
            min_kb_words: 15,
            min_prose_words: 25,
            rule_source_limit: 12,
            use_assembly: true,
            use_summarizer_fallback: true,
        }
    }
}

impl TramSettings {
    pub fn load() -> Self {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lmm")
            .join("settings.json");
        if let Ok(raw) = fs::read_to_string(&path) {
            serde_json::from_str(&raw).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save(&self) {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lmm")
            .join("settings.json");
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(&path, json);
        }
    }
}

/// Global theme configuration persisted to `~/.lmm/theme.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    pub variant: ThemeVariant,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            variant: ThemeVariant::Dark,
        }
    }
}

impl ThemeConfig {
    pub fn load() -> Self {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lmm")
            .join("theme.json");
        if let Ok(raw) = fs::read_to_string(&path) {
            serde_json::from_str(&raw).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save(&self) {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lmm")
            .join("theme.json");
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(&path, json);
        }
    }

    pub fn palette(&self) -> ThemePalette {
        ThemePalette::from_variant(self.variant)
    }
}

/// Session performance metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStats {
    pub requests: u64,
    pub responses: u64,
    pub web_chunks_ingested: usize,
    pub kb_chunks_used: usize,
    pub pipeline_duration_ms: u64,
}

/// All mutable state owned by the TUI across frames.
pub struct TuiState {
    pub active_tab: AppTab,
    pub log_lines: VecDeque<String>,
    pub thinking_lines: VecDeque<String>,
    pub stats: SessionStats,
    pub chart_data: ChartData,
    pub input_buffer: tui_input::Input,
    pub pipeline_status: String,
    pub theme_config: ThemeConfig,
    pub selected_theme_idx: usize,
    pub tram_settings: TramSettings,
    pub tram_pipeline: Option<Arc<TramPipeline>>,
    pub settings_focus_idx: usize,
    pub settings_scroll_offset: usize,
    pub settings_editing: bool,
    pub settings_edit_input: tui_input::Input,
    pub input_buffers: Vec<tui_input::Input>,
    pub log_scroll_offset: usize,
    pub log_h_scroll_offset: usize,
    pub status_message: Option<String>,
    pub should_quit: bool,
    pub tick_count: u64,
    pub last_tick_time: Instant,
    pub receiver: UnboundedReceiver<TuiEvent>,
}

impl TuiState {
    pub fn new(receiver: UnboundedReceiver<TuiEvent>) -> Self {
        let theme_config = ThemeConfig::load();
        let selected_theme_idx = crate::tui::theme::ALL_THEME_VARIANTS
            .iter()
            .position(|v| *v == theme_config.variant)
            .unwrap_or(0);

        let tram_settings = TramSettings::load();
        let tram_pipeline = Self::create_pipeline_from_settings(&tram_settings);
        let input_buffers = vec![
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
            tui_input::Input::default(),
        ];

        Self {
            active_tab: AppTab::Main,
            log_lines: VecDeque::with_capacity(500),
            thinking_lines: VecDeque::with_capacity(200),
            stats: SessionStats::default(),
            chart_data: ChartData::default(),
            input_buffer: tui_input::Input::default(),
            pipeline_status: "Idle".to_string(),
            theme_config,
            selected_theme_idx,
            tram_settings,
            tram_pipeline,
            settings_focus_idx: 0,
            settings_scroll_offset: 0,
            settings_editing: false,
            settings_edit_input: tui_input::Input::default(),
            input_buffers,
            log_scroll_offset: 0,
            log_h_scroll_offset: 0,
            status_message: None,
            should_quit: false,
            tick_count: 0,
            last_tick_time: Instant::now(),
            receiver,
        }
    }

    fn create_pipeline_from_settings(settings: &TramSettings) -> Option<Arc<TramPipeline>> {
        let config = TramConfig {
            iterations: settings.iterations,
            inference_depth: settings.inference_depth,
            cf_threshold: settings.cf_threshold,
            think: settings.think,
            think_steps: settings.think_steps,
            think_threshold: settings.think_threshold,
            think_patience: settings.think_patience,
            max_tasks: settings.max_tasks,
            probability: Some(settings.probability),
            no_frame: settings.no_frame,
            no_web: settings.no_web,
            deep: settings.deep,
            convergence_threshold: settings.convergence_threshold,
            k_proportional: settings.k_proportional,
            k_integral: settings.k_integral,
            top_k_min: settings.top_k_min,
            top_k_var: settings.top_k_var,
            min_rule_count: settings.min_rule_count,
            min_expert_words: settings.min_expert_words,
            dedup_threshold: settings.dedup_threshold,
            prose_min_sentences: settings.prose_min_sentences,
            prose_max_sentences: settings.prose_max_sentences,
            min_kb_words: settings.min_kb_words,
            min_prose_words: settings.min_prose_words,
            rule_source_limit: settings.rule_source_limit,
            use_assembly: settings.use_assembly,
            use_summarizer_fallback: settings.use_summarizer_fallback,
        };
        Some(Arc::new(TramPipeline::new(config)))
    }

    pub fn reinitialize_pipeline(&mut self) {
        self.tram_pipeline = Self::create_pipeline_from_settings(&self.tram_settings);
    }

    pub fn tick(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);
    }

    pub fn process_events(&mut self) {
        while let Ok(event) = self.receiver.try_recv() {
            match event {
                TuiEvent::Log(msg) => {
                    self.log_lines.push_back(msg);
                    if self.log_lines.len() > 500 {
                        self.log_lines.pop_front();
                    }
                }
                TuiEvent::LogAppend(msg) => {
                    if let Some(last) = self.log_lines.back_mut() {
                        last.push_str(&msg);
                    } else {
                        self.log_lines.push_back(msg);
                    }
                }
                TuiEvent::ClearLog => {
                    self.log_lines.clear();
                }
                TuiEvent::Thinking(msg) => {
                    self.thinking_lines.push_back(msg);
                    if self.thinking_lines.len() > 200 {
                        self.thinking_lines.pop_front();
                    }
                }
                TuiEvent::Response(msg) => {
                    for line in msg.lines() {
                        self.log_lines.push_back(format!("✓ {}", line));
                    }
                }
                TuiEvent::StatsUpdate(stats) => {
                    self.stats = stats.clone();
                    self.chart_data.record_stats(&stats);
                }
                TuiEvent::IncRequest => {
                    self.stats.requests += 1;
                    self.chart_data.record_stats(&self.stats);
                }
                TuiEvent::IncResponse => {
                    self.stats.responses += 1;
                    self.chart_data.record_stats(&self.stats);
                }
                TuiEvent::AddWebChunks(count) => {
                    self.stats.web_chunks_ingested += count;
                    self.chart_data.record_stats(&self.stats);
                }
                TuiEvent::AddKbChunks(count) => {
                    self.stats.kb_chunks_used += count;
                    self.chart_data.record_stats(&self.stats);
                }
                TuiEvent::PipelineStatus(status) => {
                    self.pipeline_status = status;
                }
                TuiEvent::Quit => {
                    self.should_quit = true;
                }
            }
        }
    }

    pub fn push_log(&mut self, msg: String) {
        self.log_lines.push_back(msg);
        if self.log_lines.len() > 500 {
            self.log_lines.pop_front();
        }
    }

    pub fn apply_theme_variant(&mut self, variant: ThemeVariant) {
        self.theme_config.variant = variant;
        self.theme_config.save();
        self.selected_theme_idx = crate::tui::theme::ALL_THEME_VARIANTS
            .iter()
            .position(|v| *v == variant)
            .unwrap_or(0);
    }

    pub fn save_settings(&mut self) {
        self.tram_settings.save();
        self.status_message = Some("Settings saved to ~/.lmm/settings.json".to_string());
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
