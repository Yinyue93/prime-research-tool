//! Plotting and visualization using the plotters crate

use plotters::prelude::*;
use prime_tool::gaps::GapStatistics;
use prime_tool::sieve_range;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// Generate all available plots for a given range
pub fn generate_all_plots(range: Range<u64>, output_dir: &Path) -> anyhow::Result<()> {
    println!("Generating plots for range {}..{}", range.start, range.end);
    
    // Prime counting function π(x)
    let prime_count_path = output_dir.join("prime_count.svg");
    generate_prime_count_plot(range.clone(), &prime_count_path)?;
    println!("Prime count plot saved to: {}", prime_count_path.display());
    
    // Gap analysis
    let stats = prime_tool::gaps::analyze_gaps(range.clone());
    let gap_histogram_path = output_dir.join("gap_histogram.svg");
    generate_gap_histogram(&stats, &gap_histogram_path)?;
    println!("Gap histogram saved to: {}", gap_histogram_path.display());
    
    // Prime distribution
    let distribution_path = output_dir.join("prime_distribution.svg");
    generate_distribution_plot(range.clone(), &distribution_path)?;
    println!("Prime distribution plot saved to: {}", distribution_path.display());
    
    // Gap size over range
    let gap_progression_path = output_dir.join("gap_progression.svg");
    generate_gap_progression_plot(range, &gap_progression_path)?;
    println!("Gap progression plot saved to: {}", gap_progression_path.display());
    
    Ok(())
}

/// Generate prime counting function π(x) plot
pub fn generate_prime_count_plot<P: AsRef<Path>>(
    range: Range<u64>,
    output_path: P,
) -> anyhow::Result<PathBuf> {
    let path = output_path.as_ref();
    let root = SVGBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let primes = sieve_range(range.clone());
    
    // Create data points for π(x)
    let mut data_points = Vec::new();
    let mut prime_count = 0;
    let step = ((range.end - range.start) / 1000).max(1);
    
    for x in (range.start..range.end).step_by(step as usize) {
        // Count primes up to x
        while prime_count < primes.len() && primes[prime_count] <= x {
            prime_count += 1;
        }
        data_points.push((x as f64, prime_count as f64));
    }
    
    let max_x = range.end as f64;
    let max_y = primes.len() as f64;
    
    let mut chart = ChartBuilder::on(&root)
        .caption(&format!("Prime Counting Function π(x) for {}..{}", range.start, range.end), ("Arial", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(range.start as f64..max_x, 0f64..max_y)?;
    
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("π(x)")
        .draw()?;
    
    // Plot π(x)
    chart
        .draw_series(LineSeries::new(data_points.iter().cloned(), &BLUE))?;
    
    // Add approximation x/ln(x) for comparison
    let approximation_points: Vec<(f64, f64)> = (range.start..range.end)
        .step_by(step as usize)
        .map(|x| {
            let x_f = x as f64;
            let approx = if x_f > 1.0 { x_f / x_f.ln() } else { 0.0 };
            (x_f, approx)
        })
        .collect();
    
    chart
        .draw_series(LineSeries::new(approximation_points, &RED.mix(0.7)))?;
    
    chart
        .draw_series(std::iter::once(Circle::new((0.0, 0.0), 0)))?  // Dummy for legend
        .label("π(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));
    
    chart
        .draw_series(std::iter::once(Circle::new((0.0, 0.0), 0)))?  // Dummy for legend
        .label("x/ln(x) approximation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));
    
    chart.configure_series_labels().draw()?;
    
    root.present()?;
    Ok(path.to_path_buf())
}

/// Generate gap histogram
pub fn generate_gap_histogram<P: AsRef<Path>>(
    stats: &GapStatistics,
    output_path: P,
) -> anyhow::Result<PathBuf> {
    let path = output_path.as_ref();
    let root = SVGBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut sorted_gaps: Vec<_> = stats.gap_histogram.iter().collect();
    sorted_gaps.sort_by_key(|(gap, _)| *gap);
    
    // Take top 20 most common gaps
    sorted_gaps.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    sorted_gaps.truncate(20);
    sorted_gaps.sort_by_key(|(gap, _)| *gap);
    
    let max_count = sorted_gaps.iter().map(|(_, count)| **count).max().unwrap_or(1);
    let max_gap = sorted_gaps.iter().map(|(gap, _)| **gap).max().unwrap_or(1);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Prime Gap Distribution", ("Arial", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0u32..max_gap + 1, 0usize..max_count + 1)?;
    
    chart
        .configure_mesh()
        .x_desc("Gap Size")
        .y_desc("Frequency")
        .draw()?;
    
    chart.draw_series(
        sorted_gaps
            .iter()
            .map(|(&gap, &count)| Rectangle::new([(gap, 0), (gap, count)], BLUE.filled()))
    )?;
    
    root.present()?;
    Ok(path.to_path_buf())
}

/// Generate prime distribution plot
pub fn generate_distribution_plot<P: AsRef<Path>>(
    range: Range<u64>,
    output_path: P,
) -> anyhow::Result<PathBuf> {
    let path = output_path.as_ref();
    let root = SVGBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let primes = sieve_range(range.clone());
    
    // Create histogram of prime density in intervals
    let num_intervals = 50;
    let interval_size = (range.end - range.start) / num_intervals;
    let mut intervals = vec![0; num_intervals as usize];
    
    for &prime in &primes {
        let interval_index = ((prime - range.start) / interval_size).min(num_intervals - 1) as usize;
        intervals[interval_index] += 1;
    }
    
    let max_density = *intervals.iter().max().unwrap_or(&1);
    
    let mut chart = ChartBuilder::on(&root)
        .caption(&format!("Prime Density Distribution ({}..{})", range.start, range.end), ("Arial", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(range.start..range.end, 0..max_density + 1)?;
    
    chart
        .configure_mesh()
        .x_desc("Range")
        .y_desc("Prime Count")
        .draw()?;
    
    chart.draw_series(
        intervals
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let start = range.start + i as u64 * interval_size;
                let end = start + interval_size;
                Rectangle::new([(start, 0), (end, count)], GREEN.mix(0.7).filled())
            })
    )?;
    
    root.present()?;
    Ok(path.to_path_buf())
}

/// Generate gap progression plot showing how gaps change over the range
pub fn generate_gap_progression_plot<P: AsRef<Path>>(
    range: Range<u64>,
    output_path: P,
) -> anyhow::Result<PathBuf> {
    let path = output_path.as_ref();
    let root = SVGBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let gaps = prime_tool::gaps::find_gaps(range.clone());
    
    if gaps.is_empty() {
        return Err(anyhow::anyhow!("No gaps found in range"));
    }
    
    let max_gap = gaps.iter().map(|g| g.gap_size).max().unwrap_or(1);
    
    let mut chart = ChartBuilder::on(&root)
        .caption(&format!("Prime Gap Progression ({}..{})", range.start, range.end), ("Arial", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(range.start..range.end, 0u32..max_gap + 1)?;
    
    chart
        .configure_mesh()
        .x_desc("Prime")
        .y_desc("Gap Size")
        .draw()?;
    
    // Plot gap sizes
    let gap_points: Vec<(u64, u32)> = gaps
        .iter()
        .map(|g| (g.start_prime, g.gap_size))
        .collect();
    
    chart.draw_series(LineSeries::new(gap_points.iter().cloned(), &BLUE))?;
    
    // Highlight maximal gaps
    let maximal_gaps: Vec<(u64, u32)> = gaps
        .iter()
        .filter(|g| g.is_maximal)
        .map(|g| (g.start_prime, g.gap_size))
        .collect();
    
    chart.draw_series(
        maximal_gaps
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 3, RED.filled()))
    )?;
    
    chart
        .draw_series(std::iter::once(Circle::new((0, 0), 0)))?  // Dummy for legend
        .label("Gap sizes")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));
    
    chart
        .draw_series(std::iter::once(Circle::new((0, 0), 0)))?  // Dummy for legend
        .label("Maximal gaps")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, RED.filled()));
    
    chart.configure_series_labels().draw()?;
    
    root.present()?;
    Ok(path.to_path_buf())
}

/// Generate twin prime distribution plot
pub fn generate_twin_prime_plot<P: AsRef<Path>>(
    range: Range<u64>,
    output_path: P,
) -> anyhow::Result<PathBuf> {
    let path = output_path.as_ref();
    let root = SVGBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let twin_primes = prime_tool::gaps::find_twin_primes(range.clone());
    
    // Create cumulative count of twin primes
    let mut cumulative_data = Vec::new();
    let mut count = 0;
    let step = ((range.end - range.start) / 1000).max(1);
    
    for x in (range.start..range.end).step_by(step as usize) {
        // Count twin primes up to x
        while count < twin_primes.len() && twin_primes[count].larger <= x {
            count += 1;
        }
        cumulative_data.push((x as f64, count as f64));
    }
    
    let max_count = twin_primes.len() as f64;
    
    let mut chart = ChartBuilder::on(&root)
        .caption(&format!("Twin Prime Count ({}..{})", range.start, range.end), ("Arial", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(range.start as f64..range.end as f64, 0f64..max_count + 1.0)?;
    
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("Twin Prime Count")
        .draw()?;
    
    chart.draw_series(LineSeries::new(cumulative_data, &MAGENTA))?;
    
    // Mark individual twin primes
    chart.draw_series(
        twin_primes
            .iter()
            .map(|tp| Circle::new((tp.smaller as f64, 0.0), 2, RED.filled()))
    )?;
    
    root.present()?;
    Ok(path.to_path_buf())
}

/// Generate comparative algorithm performance plot
pub fn generate_performance_plot<P: AsRef<Path>>(
    performance_data: &HashMap<String, Vec<(u64, f64)>>,
    output_path: P,
) -> anyhow::Result<PathBuf> {
    let path = output_path.as_ref();
    let root = SVGBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    if performance_data.is_empty() {
        return Err(anyhow::anyhow!("No performance data provided"));
    }
    
    let max_x = performance_data
        .values()
        .flat_map(|data| data.iter().map(|(x, _)| *x))
        .max()
        .unwrap_or(1000) as f64;
    
    let max_y = performance_data
        .values()
        .flat_map(|data| data.iter().map(|(_, y)| *y))
        .fold(0.0f64, |a, b| a.max(b));
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Algorithm Performance Comparison", ("Arial", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..max_x, 0f64..max_y * 1.1)?;
    
    chart
        .configure_mesh()
        .x_desc("Input Size")
        .y_desc("Time (ms)")
        .draw()?;
    
    let colors = [&BLUE, &RED, &GREEN, &MAGENTA, &CYAN];
    
    for (i, (algorithm, data)) in performance_data.iter().enumerate() {
        let color = colors[i % colors.len()];
        
        chart
            .draw_series(LineSeries::new(
                data.iter().map(|(x, y)| (*x as f64, *y)),
                color,
            ))?
            .label(algorithm)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
    }
    
    chart.configure_series_labels().draw()?;
    
    root.present()?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;
    
    #[test]
    fn test_generate_prime_count_plot() {
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test_prime_count.svg");
        
        let result = generate_prime_count_plot(2..100, &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
        
        let content = fs::read_to_string(&output_path).unwrap();
        assert!(content.contains("<svg"));
        assert!(content.contains("Prime Counting Function"));
    }
    
    #[test]
    fn test_generate_gap_histogram() {
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test_gaps.svg");
        
        let stats = prime_tool::gaps::analyze_gaps(2..100);
        let result = generate_gap_histogram(&stats, &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }
}