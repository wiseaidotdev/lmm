// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.

use crate::tui::state::TuiState;
use crate::tui::theme::ThemePalette;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
};

pub fn render_settings_tab(
    frame: &mut Frame,
    area: Rect,
    state: &TuiState,
    palette: &ThemePalette,
) {
    let split = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(area);

    render_toggle_settings(frame, split[0], state, palette);
    render_tram_settings(frame, split[1], state, palette);
}

fn render_toggle_settings(frame: &mut Frame, area: Rect, state: &TuiState, palette: &ThemePalette) {
    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(palette.accent)),
            Span::styled(
                "Behavior Toggles",
                Style::default().fg(palette.fg).add_modifier(Modifier::BOLD),
            ),
        ]))
        .borders(Borders::ALL)
        .border_set(symbols::border::ROUNDED)
        .border_style(Style::default().fg(palette.border))
        .style(Style::default().bg(palette.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Fill(1), Constraint::Length(5)])
        .split(inner);

    let toggles = [
        (
            "🔍 Deep Search",
            state.tram_settings.deep,
            "Enable deep web scraping",
        ),
        (
            "🧠 Think Mode",
            state.tram_settings.think,
            "Enable ATE reasoning pre-pass",
        ),
        (
            "🌐 Web Access",
            !state.tram_settings.no_web,
            "Allow web search",
        ),
        (
            "📝 Frame Response",
            !state.tram_settings.no_frame,
            "Add intent framing",
        ),
    ];

    let items: Vec<ListItem> = toggles
        .iter()
        .enumerate()
        .map(|(i, (label, enabled, description))| {
            let is_selected = state.settings_focus_idx == i;
            let (check, color) = if *enabled {
                ("[✓]", palette.ok)
            } else {
                ("[ ]", palette.muted)
            };

            let label_style = if is_selected {
                Style::default()
                    .fg(palette.tab_active_fg)
                    .bg(palette.tab_active_bg)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(palette.fg).add_modifier(Modifier::BOLD)
            };

            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(
                        format!("  {} ", check),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(label.to_string(), label_style),
                ]),
                Line::from(Span::styled(
                    format!("      {}", description),
                    Style::default().fg(palette.muted),
                )),
                Line::from(Span::raw("")),
            ])
        })
        .collect();

    frame.render_widget(List::new(items), vertical[0]);

    let hints = Paragraph::new(vec![
        Line::from(Span::styled(
            "  ↑ ↓  navigate  │  Space  toggle  │  s  save",
            Style::default().fg(palette.muted),
        )),
        Line::from(Span::styled(
            "  Changes saved to ~/.lmm/settings.json",
            Style::default().fg(palette.muted),
        )),
    ])
    .wrap(Wrap { trim: false });

    frame.render_widget(hints, vertical[1]);
}

fn render_tram_settings(frame: &mut Frame, area: Rect, state: &TuiState, palette: &ThemePalette) {
    let settings_def: Vec<(&str, String, usize)> = vec![
        (
            "Knowledge Base",
            state
                .tram_settings
                .knowledge_base
                .clone()
                .unwrap_or_else(|| "None".to_string()),
            4,
        ),
        ("Iterations", state.tram_settings.iterations.to_string(), 5),
        (
            "Inference Depth",
            state.tram_settings.inference_depth.to_string(),
            6,
        ),
        (
            "CF Threshold",
            state.tram_settings.cf_threshold.to_string(),
            7,
        ),
        (
            "Think Steps",
            state.tram_settings.think_steps.to_string(),
            8,
        ),
        (
            "Think Threshold",
            state.tram_settings.think_threshold.to_string(),
            9,
        ),
        (
            "Think Patience",
            state.tram_settings.think_patience.to_string(),
            10,
        ),
        ("Max Tasks", state.tram_settings.max_tasks.to_string(), 11),
        (
            "Probability",
            state.tram_settings.probability.to_string(),
            12,
        ),
        (
            "Conv Threshold",
            state.tram_settings.convergence_threshold.to_string(),
            13,
        ),
        (
            "K Proportional",
            state.tram_settings.k_proportional.to_string(),
            14,
        ),
        ("K Integral", state.tram_settings.k_integral.to_string(), 15),
        ("Top K Min", state.tram_settings.top_k_min.to_string(), 16),
        ("Top K Var", state.tram_settings.top_k_var.to_string(), 17),
        (
            "Min Rule Count",
            state.tram_settings.min_rule_count.to_string(),
            18,
        ),
        (
            "Min Expert Words",
            state.tram_settings.min_expert_words.to_string(),
            19,
        ),
        (
            "Dedup Threshold",
            state.tram_settings.dedup_threshold.to_string(),
            20,
        ),
        (
            "Prose Min Sent",
            state.tram_settings.prose_min_sentences.to_string(),
            21,
        ),
        (
            "Prose Max Sent",
            state.tram_settings.prose_max_sentences.to_string(),
            22,
        ),
        (
            "Min KB Words",
            state.tram_settings.min_kb_words.to_string(),
            23,
        ),
        (
            "Min Prose Words",
            state.tram_settings.min_prose_words.to_string(),
            24,
        ),
        (
            "Rule Source Limit",
            state.tram_settings.rule_source_limit.to_string(),
            25,
        ),
        (
            "Use Assembly",
            state.tram_settings.use_assembly.to_string(),
            26,
        ),
        (
            "Use Summarizer",
            state.tram_settings.use_summarizer_fallback.to_string(),
            27,
        ),
    ];

    let visible_fields = 8;
    let scroll_offset = state
        .settings_scroll_offset
        .min(settings_def.len().saturating_sub(visible_fields));
    let current_page = scroll_offset / visible_fields + 1;
    let total_pages = settings_def.len().div_ceil(visible_fields);
    let scroll_indicator = format!(" [{}/{}]", current_page, total_pages);
    let edit_indicator = if state.settings_editing {
        " [EDITING]"
    } else {
        ""
    };

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(palette.accent)),
            Span::styled(
                "TRAM Configuration",
                Style::default().fg(palette.fg).add_modifier(Modifier::BOLD),
            ),
            Span::styled(scroll_indicator, Style::default().fg(palette.muted)),
            Span::styled(
                edit_indicator,
                Style::default()
                    .fg(palette.warn)
                    .add_modifier(Modifier::BOLD),
            ),
        ]))
        .borders(Borders::ALL)
        .border_set(symbols::border::ROUNDED)
        .border_style(Style::default().fg(palette.border))
        .style(Style::default().bg(palette.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let visible_settings: Vec<_> = settings_def
        .iter()
        .skip(scroll_offset)
        .take(visible_fields)
        .collect();

    let mut all_lines: Vec<Line> = vec![Line::from(Span::raw(""))];

    for (label, value, idx) in &visible_settings {
        let is_selected = state.settings_focus_idx == *idx;
        let is_editing = state.settings_editing && is_selected;

        all_lines.push(Line::from(Span::styled(
            format!("  {}", label),
            Style::default().fg(palette.muted),
        )));

        if is_editing {
            let input_text = state.settings_edit_input.value();
            all_lines.push(Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(
                    input_text,
                    Style::default()
                        .fg(palette.accent)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        } else {
            let val_str = value;
            let val_style = if is_selected {
                Style::default()
                    .fg(palette.tab_active_fg)
                    .bg(palette.tab_active_bg)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(palette.fg)
            };
            all_lines.push(Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(val_str.to_string(), val_style),
            ]));
        }

        all_lines.push(Line::from(Span::styled(
            "  ───────────────────────────────────────",
            Style::default().fg(palette.border),
        )));
    }

    frame.render_widget(Paragraph::new(all_lines).wrap(Wrap { trim: true }), inner);

    if state.settings_editing {
        let input_cursor = state.settings_edit_input.visual_cursor();
        let focused_idx = visible_settings
            .iter()
            .position(|(_, _, idx)| *idx == state.settings_focus_idx)
            .unwrap_or(0);
        let cursor_y = inner.y + (focused_idx * 3 + 2) as u16;
        let cursor_x = inner.x + input_cursor as u16;
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.
