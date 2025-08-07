//! Output formatting for CLI results

use comfy_table::{Table, Cell, Attribute, Color};
use prime_tool::gaps::{GapStatistics, TwinPrime};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Output format for primality test results
#[derive(Debug, Serialize, Deserialize)]
pub struct PrimalityOutput {
    pub number: u64,
    pub is_prime: bool,
    pub execution_time_ms: u64,
}

/// Output format for factorization results
#[derive(Debug, Serialize, Deserialize)]
pub struct FactorizationOutput {
    pub number: u64,
    pub factors: Vec<u64>,
    pub execution_time_ms: u64,
}

/// Output format for sieve results
#[derive(Debug, Serialize, Deserialize)]
pub struct SieveOutput {
    pub range_start: u64,
    pub range_end: u64,
    pub primes: Vec<u64>,
    pub execution_time_ms: u64,
}

/// Output format for gap analysis results
#[derive(Debug, Serialize, Deserialize)]
pub struct GapAnalysisOutput {
    pub range_start: u64,
    pub range_end: u64,
    pub statistics: GapStatistics,
    pub twin_primes: Option<Vec<TwinPrime>>,
    pub cousin_primes: Option<Vec<TwinPrime>>,
    pub sexy_primes: Option<Vec<TwinPrime>>,
    pub execution_time_ms: u64,
}

/// Print output in the specified format
pub fn print_output<T: Serialize + OutputDisplay>(
    data: &T,
    format: &crate::OutputFormat,
    output_file: Option<&Path>,
) -> anyhow::Result<()> {
    let content = match format {
        crate::OutputFormat::Table => data.format_table(),
        crate::OutputFormat::Json => serde_json::to_string_pretty(data)?,
        crate::OutputFormat::Csv => data.format_csv()?,
        crate::OutputFormat::Plain => data.format_plain(),
    };
    
    if let Some(path) = output_file {
        let mut file = File::create(path)?;
        file.write_all(content.as_bytes())?;
        println!("Output written to: {}", path.display());
    } else {
        println!("{}", content);
    }
    
    Ok(())
}

/// Trait for custom output formatting
pub trait OutputDisplay {
    fn format_table(&self) -> String;
    fn format_csv(&self) -> anyhow::Result<String>;
    fn format_plain(&self) -> String;
}

impl OutputDisplay for PrimalityOutput {
    fn format_table(&self) -> String {
        let mut table = Table::new();
        table.set_header(vec!["Property", "Value"]);
        
        table.add_row(vec![
            Cell::new("Number").add_attribute(Attribute::Bold),
            Cell::new(&self.number.to_string()),
        ]);
        
        let is_prime_cell = if self.is_prime {
            Cell::new("PRIME").fg(Color::Green).add_attribute(Attribute::Bold)
        } else {
            Cell::new("COMPOSITE").fg(Color::Red).add_attribute(Attribute::Bold)
        };
        
        table.add_row(vec![
            Cell::new("Result").add_attribute(Attribute::Bold),
            is_prime_cell,
        ]);
        
        table.add_row(vec![
            Cell::new("Execution Time").add_attribute(Attribute::Bold),
            Cell::new(&format!("{} ms", self.execution_time_ms)),
        ]);
        
        table.to_string()
    }
    
    fn format_csv(&self) -> anyhow::Result<String> {
        let mut wtr = csv::Writer::from_writer(vec![]);
        wtr.write_record(&["number", "is_prime", "execution_time_ms"])?;
        wtr.write_record(&[
            &self.number.to_string(),
            &self.is_prime.to_string(),
            &self.execution_time_ms.to_string(),
        ])?;
        Ok(String::from_utf8(wtr.into_inner()?)?)
    }
    
    fn format_plain(&self) -> String {
        format!(
            "{} is {}\nExecution time: {} ms",
            self.number,
            if self.is_prime { "PRIME" } else { "COMPOSITE" },
            self.execution_time_ms
        )
    }
}

impl OutputDisplay for FactorizationOutput {
    fn format_table(&self) -> String {
        let mut table = Table::new();
        table.set_header(vec!["Property", "Value"]);
        
        table.add_row(vec![
            Cell::new("Number").add_attribute(Attribute::Bold),
            Cell::new(&self.number.to_string()),
        ]);
        
        let factors_str = if self.factors.len() == 1 {
            "PRIME".to_string()
        } else {
            format_factors(&self.factors)
        };
        
        table.add_row(vec![
            Cell::new("Factorization").add_attribute(Attribute::Bold),
            Cell::new(&factors_str),
        ]);
        
        table.add_row(vec![
            Cell::new("Execution Time").add_attribute(Attribute::Bold),
            Cell::new(&format!("{} ms", self.execution_time_ms)),
        ]);
        
        table.to_string()
    }
    
    fn format_csv(&self) -> anyhow::Result<String> {
        let mut wtr = csv::Writer::from_writer(vec![]);
        wtr.write_record(&["number", "factors", "execution_time_ms"])?;
        wtr.write_record(&[
            &self.number.to_string(),
            &format!("{:?}", self.factors),
            &self.execution_time_ms.to_string(),
        ])?;
        Ok(String::from_utf8(wtr.into_inner()?)?)
    }
    
    fn format_plain(&self) -> String {
        format!(
            "{} = {}\nExecution time: {} ms",
            self.number,
            format_factors(&self.factors),
            self.execution_time_ms
        )
    }
}

impl OutputDisplay for SieveOutput {
    fn format_table(&self) -> String {
        let mut table = Table::new();
        table.set_header(vec!["Property", "Value"]);
        
        table.add_row(vec![
            Cell::new("Range").add_attribute(Attribute::Bold),
            Cell::new(&format!("{}..{}", self.range_start, self.range_end)),
        ]);
        
        table.add_row(vec![
            Cell::new("Primes Found").add_attribute(Attribute::Bold),
            Cell::new(&self.primes.len().to_string()).fg(Color::Green),
        ]);
        
        table.add_row(vec![
            Cell::new("Execution Time").add_attribute(Attribute::Bold),
            Cell::new(&format!("{} ms", self.execution_time_ms)),
        ]);
        
        if self.primes.len() <= 100 {
            table.add_row(vec![
                Cell::new("Primes").add_attribute(Attribute::Bold),
                Cell::new(&format!("{:?}", self.primes)),
            ]);
        } else {
            let first_10: Vec<String> = self.primes.iter().take(10).map(|p| p.to_string()).collect();
            let last_10: Vec<String> = self.primes.iter().rev().take(10).rev().map(|p| p.to_string()).collect();
            table.add_row(vec![
                Cell::new("First 10 Primes").add_attribute(Attribute::Bold),
                Cell::new(&first_10.join(", ")),
            ]);
            table.add_row(vec![
                Cell::new("Last 10 Primes").add_attribute(Attribute::Bold),
                Cell::new(&last_10.join(", ")),
            ]);
        }
        
        table.to_string()
    }
    
    fn format_csv(&self) -> anyhow::Result<String> {
        let mut wtr = csv::Writer::from_writer(vec![]);
        wtr.write_record(&["prime"])?;
        for &prime in &self.primes {
            wtr.write_record(&[&prime.to_string()])?;
        }
        Ok(String::from_utf8(wtr.into_inner()?)?)
    }
    
    fn format_plain(&self) -> String {
        let mut result = format!(
            "Range: {}..{}\nPrimes found: {}\nExecution time: {} ms\n",
            self.range_start, self.range_end, self.primes.len(), self.execution_time_ms
        );
        
        if self.primes.len() <= 100 {
            result.push_str(&format!("Primes: {:?}\n", self.primes));
        } else {
            let first_10: Vec<String> = self.primes.iter().take(10).map(|p| p.to_string()).collect();
            let last_10: Vec<String> = self.primes.iter().rev().take(10).rev().map(|p| p.to_string()).collect();
            result.push_str(&format!("First 10: {}\n", first_10.join(", ")));
            result.push_str(&format!("Last 10: {}\n", last_10.join(", ")));
        }
        
        result
    }
}

impl OutputDisplay for GapAnalysisOutput {
    fn format_table(&self) -> String {
        let mut table = Table::new();
        table.set_header(vec!["Property", "Value"]);
        
        table.add_row(vec![
            Cell::new("Range").add_attribute(Attribute::Bold),
            Cell::new(&format!("{}..{}", self.range_start, self.range_end)),
        ]);
        
        table.add_row(vec![
            Cell::new("Total Gaps").add_attribute(Attribute::Bold),
            Cell::new(&self.statistics.total_gaps.to_string()),
        ]);
        
        table.add_row(vec![
            Cell::new("Min Gap").add_attribute(Attribute::Bold),
            Cell::new(&self.statistics.min_gap.to_string()),
        ]);
        
        table.add_row(vec![
            Cell::new("Max Gap").add_attribute(Attribute::Bold),
            Cell::new(&self.statistics.max_gap.to_string()).fg(Color::Red),
        ]);
        
        table.add_row(vec![
            Cell::new("Average Gap").add_attribute(Attribute::Bold),
            Cell::new(&format!("{:.2}", self.statistics.average_gap)),
        ]);
        
        table.add_row(vec![
            Cell::new("Median Gap").add_attribute(Attribute::Bold),
            Cell::new(&format!("{:.2}", self.statistics.median_gap)),
        ]);
        
        table.add_row(vec![
            Cell::new("Twin Primes").add_attribute(Attribute::Bold),
            Cell::new(&self.statistics.twin_prime_count.to_string()).fg(Color::Green),
        ]);
        
        table.add_row(vec![
            Cell::new("Cousin Primes").add_attribute(Attribute::Bold),
            Cell::new(&self.statistics.cousin_prime_count.to_string()).fg(Color::Blue),
        ]);
        
        table.add_row(vec![
            Cell::new("Sexy Primes").add_attribute(Attribute::Bold),
            Cell::new(&self.statistics.sexy_prime_count.to_string()).fg(Color::Magenta),
        ]);
        
        table.add_row(vec![
            Cell::new("Execution Time").add_attribute(Attribute::Bold),
            Cell::new(&format!("{} ms", self.execution_time_ms)),
        ]);
        
        // Add gap histogram summary
        let mut histogram_str = String::new();
        let mut sorted_gaps: Vec<_> = self.statistics.gap_histogram.iter().collect();
        sorted_gaps.sort_by_key(|(gap, _)| *gap);
        
        for (gap, count) in sorted_gaps.iter().take(10) {
            histogram_str.push_str(&format!("{}:{}, ", gap, count));
        }
        if histogram_str.len() > 2 {
            histogram_str.truncate(histogram_str.len() - 2);
        }
        
        table.add_row(vec![
            Cell::new("Top Gap Sizes").add_attribute(Attribute::Bold),
            Cell::new(&histogram_str),
        ]);
        
        table.to_string()
    }
    
    fn format_csv(&self) -> anyhow::Result<String> {
        let mut wtr = csv::Writer::from_writer(vec![]);
        
        // Write statistics
        wtr.write_record(&[
            "range_start", "range_end", "total_gaps", "min_gap", "max_gap",
            "average_gap", "median_gap", "twin_primes", "cousin_primes", "sexy_primes",
            "execution_time_ms"
        ])?;
        
        wtr.write_record(&[
            &self.range_start.to_string(),
            &self.range_end.to_string(),
            &self.statistics.total_gaps.to_string(),
            &self.statistics.min_gap.to_string(),
            &self.statistics.max_gap.to_string(),
            &format!("{:.2}", self.statistics.average_gap),
            &format!("{:.2}", self.statistics.median_gap),
            &self.statistics.twin_prime_count.to_string(),
            &self.statistics.cousin_prime_count.to_string(),
            &self.statistics.sexy_prime_count.to_string(),
            &self.execution_time_ms.to_string(),
        ])?;
        
        Ok(String::from_utf8(wtr.into_inner()?)?)
    }
    
    fn format_plain(&self) -> String {
        let mut result = format!(
            "Gap Analysis for Range {}..{}\n",
            self.range_start, self.range_end
        );
        
        result.push_str(&format!("Total gaps: {}\n", self.statistics.total_gaps));
        result.push_str(&format!("Min gap: {}\n", self.statistics.min_gap));
        result.push_str(&format!("Max gap: {}\n", self.statistics.max_gap));
        result.push_str(&format!("Average gap: {:.2}\n", self.statistics.average_gap));
        result.push_str(&format!("Median gap: {:.2}\n", self.statistics.median_gap));
        result.push_str(&format!("Twin primes: {}\n", self.statistics.twin_prime_count));
        result.push_str(&format!("Cousin primes: {}\n", self.statistics.cousin_prime_count));
        result.push_str(&format!("Sexy primes: {}\n", self.statistics.sexy_prime_count));
        result.push_str(&format!("Execution time: {} ms\n", self.execution_time_ms));
        
        if let Some(ref twins) = self.twin_primes {
            result.push_str(&format!("\nTwin primes found: {}\n", twins.len()));
            if twins.len() <= 20 {
                for twin in twins {
                    result.push_str(&format!("({}, {})\n", twin.smaller, twin.larger));
                }
            }
        }
        
        result
    }
}

/// Format factors with proper mathematical notation
fn format_factors(factors: &[u64]) -> String {
    if factors.is_empty() {
        return "1".to_string();
    }
    
    if factors.len() == 1 {
        return factors[0].to_string();
    }
    
    // Group factors and show with exponents
    let mut factor_counts: HashMap<u64, usize> = HashMap::new();
    for &factor in factors {
        *factor_counts.entry(factor).or_insert(0) += 1;
    }
    
    let mut sorted_factors: Vec<_> = factor_counts.iter().collect();
    sorted_factors.sort_by_key(|(factor, _)| *factor);
    
    let formatted: Vec<String> = sorted_factors
        .iter()
        .map(|(&factor, &count)| {
            if count == 1 {
                factor.to_string()
            } else {
                format!("{}^{}", factor, count)
            }
        })
        .collect();
    
    formatted.join(" × ")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_factors() {
        assert_eq!(format_factors(&[2, 2, 3, 5]), "2^2 × 3 × 5");
        assert_eq!(format_factors(&[7]), "7");
        assert_eq!(format_factors(&[2, 3, 5, 7]), "2 × 3 × 5 × 7");
    }
    
    #[test]
    fn test_primality_output() {
        let output = PrimalityOutput {
            number: 97,
            is_prime: true,
            execution_time_ms: 1,
        };
        
        let plain = output.format_plain();
        assert!(plain.contains("97 is PRIME"));
        assert!(plain.contains("1 ms"));
    }
}