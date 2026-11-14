// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::tui::state::{AppTab, TuiState};
use crate::tui::theme::ThemePalette;
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs, Wrap},
};
use unicode_width::UnicodeWidthStr;

pub fn render_tab_bar(frame: &mut Frame, area: Rect, state: &TuiState, palette: &ThemePalette) {
    let tabs: Vec<Line> = AppTab::all()
        .into_iter()
        .map(|tab| {
            let is_active = tab == state.active_tab;
            let style = if is_active {
                Style::default()
                    .fg(palette.tab_active_fg)
                    .bg(palette.tab_active_bg)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(palette.muted)
            };
            Line::from(Span::styled(tab.title(), style))
        })
        .collect();

    let active_tab_idx = AppTab::all()
        .iter()
        .position(|t| *t == state.active_tab)
        .unwrap_or(0);

    let tab_widget = Tabs::new(tabs)
        .select(active_tab_idx)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(palette.border))
                .style(Style::default().bg(palette.bg)),
        )
        .highlight_style(
            Style::default()
                .fg(palette.tab_active_fg)
                .bg(palette.tab_active_bg)
                .add_modifier(Modifier::BOLD),
        )
        .divider(Span::styled(" │ ", Style::default().fg(palette.border)));

    frame.render_widget(tab_widget, area);
}

pub fn render_status_bar(frame: &mut Frame, area: Rect, state: &TuiState, palette: &ThemePalette) {
    let status_color = match state.pipeline_status.as_str() {
        "Idle" => palette.muted,
        "Searching..." | "Decomposing..." => palette.warn,
        "Thinking..." | "Reasoning..." => palette.accent,
        "Retrieving..." | "Inferring..." => palette.chart_2,
        "Enhancing..." | "Framing..." | "Composing..." => palette.chart_1,
        "Done" => palette.ok,
        _ => palette.fg,
    };

    let web_icon = if state.tram_settings.no_web {
        "✗web"
    } else {
        "🌐"
    };
    let deep_badge = if state.tram_settings.deep {
        " 🔍"
    } else {
        ""
    };
    let think_badge = if state.tram_settings.think {
        " 🧠"
    } else {
        ""
    };

    let spinner_icon = if state.pipeline_status == "Idle" {
        " ❯ ".to_string()
    } else {
        let chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let idx = (state.tick_count as usize / 2) % chars.len();
        format!(" {} ", chars[idx])
    };

    let left = Line::from(vec![
        Span::styled(
            spinner_icon,
            Style::default()
                .fg(palette.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{}{}{}", state.pipeline_status, deep_badge, think_badge),
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  │  ", Style::default().fg(palette.border)),
        Span::styled(web_icon, Style::default().fg(palette.muted)),
        Span::styled("  │  ", Style::default().fg(palette.border)),
        Span::styled(
            format!(
                "↑{}req  ↓{}res  ~{}web  ~{}kb",
                state.stats.requests,
                state.stats.responses,
                state.stats.web_chunks_ingested,
                state.stats.kb_chunks_used,
            ),
            Style::default().fg(palette.muted),
        ),
    ]);

    let help_text = "  Tab/Shift+Tab: switch tabs  │  q: quit  ";
    let right = Line::from(Span::styled(help_text, Style::default().fg(palette.muted)));

    let bar_block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(palette.border))
        .style(Style::default().bg(palette.bg));

    let inner = bar_block.inner(area);
    frame.render_widget(bar_block, area);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Fill(1), Constraint::Min(60)])
        .split(inner);

    frame.render_widget(Paragraph::new(left), cols[0]);
    frame.render_widget(Paragraph::new(right).alignment(Alignment::Right), cols[1]);
}

pub fn render_logo(palette: &ThemePalette) -> Vec<Line<'static>> {
    let logo_lines = [
        "  ▄▄▄      ▄▄▄     ▄▄▄   ▄▄▄     ▄▄▄  ",
        " ▀██▀       ███▄ ▄███     ███▄ ▄███   ",
        "  ██        ██ ▀█▀ ██     ██ ▀█▀ ██   ",
        "  ██        ██     ██     ██     ██   ",
        "  ██        ██     ██     ██     ██   ",
        " ████████ ▀██▀     ▀██▄ ▀██▀     ▀██▄",
    ];
    logo_lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let (r, g, b) = palette.logo_gradient[i];
            Line::from(Span::styled(
                line.to_string(),
                Style::default()
                    .fg(ratatui::style::Color::Rgb(r, g, b))
                    .add_modifier(Modifier::BOLD),
            ))
        })
        .collect()
}

pub fn render_main_tab(frame: &mut Frame, area: Rect, state: &TuiState, palette: &ThemePalette) {
    let constraints = [
        Constraint::Length(8),
        Constraint::Fill(1),
        Constraint::Length(3),
    ];

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let top_split = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(40), Constraint::Fill(1)])
        .split(vertical[0]);

    let logo_block = Block::default()
        .borders(Borders::NONE)
        .style(Style::default().bg(palette.bg));
    let logo_inner = logo_block.inner(top_split[0]);
    frame.render_widget(logo_block, top_split[0]);

    let logo_lines = render_logo(palette);
    frame.render_widget(
        Paragraph::new(logo_lines).alignment(Alignment::Left),
        logo_inner,
    );

    render_stats_pane(frame, top_split[1], state, palette);

    render_log_pane(frame, vertical[1], state, palette);

    let input_area = vertical[2];
    render_input_bar(frame, input_area, state, palette);
}

fn render_stats_pane(frame: &mut Frame, area: Rect, state: &TuiState, palette: &ThemePalette) {
    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(palette.accent)),
            Span::styled(
                "Session Stats",
                Style::default().fg(palette.fg).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" ", Style::default()),
        ]))
        .borders(Borders::ALL)
        .border_set(symbols::border::ROUNDED)
        .border_style(Style::default().fg(palette.border))
        .style(Style::default().bg(palette.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = vec![
        Line::from(vec![
            Span::styled("  Search Engine ", Style::default().fg(palette.muted)),
            Span::styled(
                "DuckDuckGo",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Mode  ", Style::default().fg(palette.muted)),
            Span::styled("TRAM", Style::default().fg(palette.chart_1)),
        ]),
        Line::from(vec![
            Span::styled("  Requests     ", Style::default().fg(palette.muted)),
            Span::styled(
                state.stats.requests.to_string(),
                Style::default().fg(palette.fg),
            ),
            Span::styled("  Responses    ", Style::default().fg(palette.muted)),
            Span::styled(
                state.stats.responses.to_string(),
                Style::default().fg(palette.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Web Chunks   ", Style::default().fg(palette.muted)),
            Span::styled(
                state.stats.web_chunks_ingested.to_string(),
                Style::default().fg(palette.chart_1),
            ),
            Span::styled("  KB Chunks    ", Style::default().fg(palette.muted)),
            Span::styled(
                state.stats.kb_chunks_used.to_string(),
                Style::default().fg(palette.chart_2),
            ),
        ]),
    ];

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn render_log_pane(frame: &mut Frame, area: Rect, state: &TuiState, palette: &ThemePalette) {
    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(palette.accent)),
            Span::styled(
                "Activity Log",
                Style::default().fg(palette.fg).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "  ↑/↓ scroll  ←/→ horizontal ",
                Style::default().fg(palette.muted),
            ),
        ]))
        .borders(Borders::ALL)
        .border_set(symbols::border::ROUNDED)
        .border_style(Style::default().fg(palette.border))
        .style(Style::default().bg(palette.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let log_lines: Vec<Line> = state
        .log_lines
        .iter()
        .map(|s| Line::from(Span::styled(s.as_str(), Style::default().fg(palette.fg))))
        .collect();

    let paragraph = Paragraph::new(log_lines)
        .wrap(Wrap { trim: false })
        .scroll((
            state.log_scroll_offset as u16,
            state.log_h_scroll_offset as u16,
        ));

    frame.render_widget(paragraph, inner);
}

fn render_input_bar(frame: &mut Frame, area: Rect, state: &TuiState, palette: &ThemePalette) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(symbols::border::ROUNDED)
        .border_style(Style::default().fg(palette.accent))
        .style(Style::default().bg(palette.input_bg))
        .title(Line::from(vec![Span::styled(
            " Request ",
            Style::default().fg(palette.fg).add_modifier(Modifier::BOLD),
        )]))
        .title_alignment(Alignment::Left);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let input_text = state.input_buffer.value();
    let cursor_pos = state.input_buffer.visual_cursor();
    let display_width = inner.width as usize;
    let start = cursor_pos.saturating_sub(display_width.saturating_sub(3));
    let visible: String = input_text
        .chars()
        .skip(start)
        .take(display_width - 2)
        .collect();

    let placeholder_visible = visible.is_empty();
    let line = if placeholder_visible {
        Line::from(vec![
            Span::styled(
                "❯ ",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Type your request...", Style::default().fg(palette.muted)),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                "❯ ",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(visible.clone(), Style::default().fg(palette.fg)),
        ])
    };

    frame.render_widget(Paragraph::new(line), inner);

    let input_prefix = "\u{276F} ";
    let prefix_width = UnicodeWidthStr::width(input_prefix) as u16;
    let cursor_x = inner.x + prefix_width + (cursor_pos - start).min(display_width - 3) as u16;
    frame.set_cursor_position((cursor_x, inner.y));
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to these terms.
