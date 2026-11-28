// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
//! Shared Plotters chart helpers for the TRAM benchmark suite.

use plotters::prelude::*;
use std::path::Path;

/// Accent colour used throughout the TRAM benchmark charts.
pub const TRAM_BLUE: RGBColor = RGBColor(37, 99, 235);
/// Background fill colour.
pub const CHART_BG: RGBColor = WHITE;

/// Saves a per-iteration latency line chart (µs) with mean and median overlays.
///
/// Used by every stage benchmark to show the raw latency distribution
/// across iterations.
pub fn plot_latency_chart(
    title: &str,
    y_label: &str,
    latencies: &[u64],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = std::fs::create_dir_all(Path::new(path).parent().unwrap_or(Path::new(".")));
    let root = BitMapBackend::new(path, (1200, 700)).into_drawing_area();
    root.fill(&CHART_BG)?;

    let max_v = *latencies.iter().max().unwrap_or(&1).max(&1);
    let min_v = *latencies.iter().min().unwrap_or(&0);

    let mean = latencies.iter().copied().sum::<u64>() as f64 / latencies.len() as f64;
    let median = {
        let mut s = latencies.to_vec();
        s.sort_unstable();
        let m = s.len() / 2;
        if s.len() % 2 == 0 {
            (s[m - 1] + s[m]) as f64 / 2.0
        } else {
            s[m] as f64
        }
    };

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 28).into_font())
        .margin(40)
        .x_label_area_size(44)
        .y_label_area_size(70)
        .build_cartesian_2d(0..latencies.len(), min_v..max_v)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 18))
        .light_line_style(TRANSPARENT)
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            latencies.iter().enumerate().map(|(i, &v)| (i, v)),
            &RED,
        ))?
        .label("Latency")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED));

    chart
        .draw_series(std::iter::once(PathElement::new(
            [(0, mean as u64), (latencies.len(), mean as u64)],
            ShapeStyle {
                color: TRAM_BLUE.mix(0.8).to_rgba(),
                filled: false,
                stroke_width: 2,
            },
        )))?
        .label(format!("Mean {mean:.1} µs"))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], TRAM_BLUE));

    chart
        .draw_series(std::iter::once(PathElement::new(
            [(0, median as u64), (latencies.len(), median as u64)],
            ShapeStyle {
                color: GREEN.mix(0.8).to_rgba(),
                filled: false,
                stroke_width: 2,
            },
        )))?
        .label(format!("Median {median:.1} µs"))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], GREEN));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 16))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Saves a grouped bar chart for multi-series comparison (e.g. TRAM vs LLMs).
///
/// `series` is a list of `(series_name, colour, values)` where values is
/// `Vec<(x_category_label, y_value)>`.
pub fn plot_bar_chart(
    title: &str,
    x_label: &str,
    y_label: &str,
    bars: &[(&str, RGBColor, f64)],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = std::fs::create_dir_all(Path::new(path).parent().unwrap_or(Path::new(".")));
    let root = BitMapBackend::new(path, (1200, 700)).into_drawing_area();
    root.fill(&CHART_BG)?;

    let max_val = bars.iter().map(|(_, _, v)| *v).fold(0.0_f64, f64::max) * 1.15;
    let n = bars.len();

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 28).into_font())
        .margin(40)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0..n, 0.0..max_val)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    for (i, (label, colour, value)) in bars.iter().enumerate() {
        let col = *colour;
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(i, 0.0), (i + 1, *value)],
                col.mix(0.85).filled(),
            )))?
            .label(*label)
            .legend(move |(x, y)| Rectangle::new([(x, y - 6), (x + 18, y + 6)], col.filled()));

        chart.draw_series(std::iter::once(Text::new(
            format!("{:.1}", value),
            (i, *value * 1.02),
            ("sans-serif", 14).into_font(),
        )))?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 15))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Saves a line/scatter throughput chart (ops/sec or words/sec) across input sizes.
pub fn plot_throughput_chart(
    title: &str,
    x_label: &str,
    y_label: &str,
    sizes: &[usize],
    throughputs: &[f64],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = std::fs::create_dir_all(Path::new(path).parent().unwrap_or(Path::new(".")));
    let root = BitMapBackend::new(path, (1200, 700)).into_drawing_area();
    root.fill(&CHART_BG)?;

    let max_x = *sizes.iter().max().unwrap_or(&1);
    let max_y = throughputs.iter().copied().fold(0.0_f64, f64::max) * 1.15;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 28).into_font())
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(0..max_x, 0.0..max_y)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 18))
        .light_line_style(TRANSPARENT)
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            sizes.iter().zip(throughputs.iter()).map(|(&x, &y)| (x, y)),
            TRAM_BLUE.stroke_width(3),
        ))?
        .label("TRAM")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], TRAM_BLUE));

    chart.draw_series(
        sizes
            .iter()
            .zip(throughputs.iter())
            .map(|(&x, &y)| Circle::new((x, y), 5, TRAM_BLUE.filled())),
    )?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 16))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Saves a latency-vs-input-size line chart showing O(n) or O(n²) growth.
pub fn plot_scaling_chart(
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[(&str, RGBColor, Vec<(usize, f64)>)],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = std::fs::create_dir_all(Path::new(path).parent().unwrap_or(Path::new(".")));
    let root = BitMapBackend::new(path, (1200, 700)).into_drawing_area();
    root.fill(&CHART_BG)?;

    let max_x = series
        .iter()
        .flat_map(|(_, _, pts)| pts.iter().map(|&(x, _)| x))
        .max()
        .unwrap_or(1);
    let max_y = series
        .iter()
        .flat_map(|(_, _, pts)| pts.iter().map(|&(_, y)| y))
        .fold(0.0_f64, f64::max)
        * 1.15;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 28).into_font())
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(0..max_x, 0.0..max_y)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 18))
        .light_line_style(TRANSPARENT)
        .draw()?;

    for (label, colour, points) in series {
        let col = *colour;
        chart
            .draw_series(LineSeries::new(points.iter().copied(), col.stroke_width(3)))?
            .label(*label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], col));
        chart.draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 5, col.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 16))
        .draw()?;

    root.present()?;
    Ok(())
}
