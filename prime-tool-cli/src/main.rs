//! Prime Research Tool CLI
//!
//! A comprehensive command-line interface for prime number research with
//! advanced algorithms, parallel processing, and visualization capabilities.

use clap::{Parser, Subcommand};
use prime_tool::*;
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

mod output;
mod plotting;
mod benchmarks;

use output::*;
use plotting::*;

#[derive(Parser)]
#[command(name = "prime-tool")]
#[command(about = "A high-performance prime research tool with advanced algorithms")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Number of threads to use (default: auto-detect)
    #[arg(short = 'j', long)]
    threads: Option<usize>,
    
    /// Output format
    #[arg(short, long, value_enum, default_value = "table")]
    format: OutputFormat,
    
    /// Output file path
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Test if a number is prime
    IsPrime {
        /// Number to test
        #[arg(short, long)]
        number: u64,
    },
    
    /// Factor a number into prime components
    Factor {
        /// Number to factor
        #[arg(short, long)]
        number: u64,
    },
    
    /// Find all primes in a range
    Sieve {
        /// Range to sieve (e.g., "1..1000000")
        #[arg(short, long)]
        range: String,
        
        /// Enable parallel processing
        #[arg(short, long)]
        parallel: bool,
        
        /// Chunk size for parallel processing
        #[arg(short, long, default_value = "10000")]
        chunk_size: usize,
        
        /// Network nodes for distributed processing
        #[arg(short, long)]
        nodes: Vec<String>,
    },
    
    /// Analyze prime gaps in a range
    Gaps {
        /// Range to analyze
        #[arg(short, long)]
        range: String,
        
        /// Find twin primes
        #[arg(short, long)]
        twins: bool,
        
        /// Find cousin primes (gap of 4)
        #[arg(short, long)]
        cousins: bool,
        
        /// Find sexy primes (gap of 6)
        #[arg(short, long)]
        sexy: bool,
        
        /// Generate gap histogram plot
        #[arg(short = 'p', long)]
        plot: bool,
    },
    
    /// Generate plots and visualizations
    Plot {
        /// Range for prime counting function π(x)
        #[arg(short, long)]
        range: String,
        
        /// Output directory for plots
        #[arg(short, long, default_value = "plots")]
        output_dir: PathBuf,
        
        /// Plot types to generate
        #[arg(short, long, value_enum, default_value = "all")]
        plot_type: PlotType,
    },
    
    /// Run benchmarks
    Benchmark {
        /// Benchmark type
        #[arg(short, long, value_enum, default_value = "all")]
        bench_type: BenchmarkType,
        
        /// Maximum number to test
        #[arg(short, long, default_value = "100000000")]
        max_n: u64,
        
        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,
    },
    
    /// Start web server
    Web {
        /// Port to bind to
        #[arg(short, long, default_value = "3000")]
        port: u16,
        
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,
    },
}

#[derive(clap::ValueEnum, Clone)]
enum OutputFormat {
    Table,
    Json,
    Csv,
    Plain,
}

#[derive(clap::ValueEnum, Clone)]
enum PlotType {
    All,
    PrimeCount,
    Gaps,
    Distribution,
}

#[derive(clap::ValueEnum, Clone)]
enum BenchmarkType {
    All,
    Primality,
    Factorization,
    Sieve,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    if cli.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }
    
    // Set thread pool size
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }
    
    let start_time = Instant::now();
    
    match cli.command {
        Commands::IsPrime { number } => {
            let result = is_prime(number);
            let output = PrimalityOutput {
                number,
                is_prime: result,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
            };
            print_output(&output, &cli.format, cli.output.as_ref().map(|v| &**v))?;
        }
        
        Commands::Factor { number } => {
            let factors = factor(number);
            let output = FactorizationOutput {
                number,
                factors,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
            };
            print_output(&output, &cli.format, cli.output.as_ref().map(|v| &**v))?;
        }
        
        Commands::Sieve { range, parallel, chunk_size, nodes } => {
            let range = parse_range(&range)?;
            let range_start = range.start;
            let range_end = range.end;
            
            let primes = if !nodes.is_empty() {
                // Distributed sieving
                let sieve = prime_tool::sieve::DistributedSieve::new(nodes, chunk_size as u64);
                sieve.sieve_distributed(range).await?
            } else if parallel {
                // Parallel sieving
                sieve_parallel(range, chunk_size)
            } else {
                // Sequential sieving
                sieve_range(range)
            };
            
            let output = SieveOutput {
                range_start,
                range_end,
                primes,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
            };
            print_output(&output, &cli.format, cli.output.as_ref().map(|v| &**v))?;
        }
        
        Commands::Gaps { range, twins, cousins, sexy, plot } => {
            let range = parse_range(&range)?;
            let stats = gaps::analyze_gaps(range.clone());
            
            let output = GapAnalysisOutput {
                range_start: range.start,
                range_end: range.end,
                statistics: stats,
                twin_primes: if twins { Some(gaps::find_twin_primes(range.clone())) } else { None },
                cousin_primes: if cousins { Some(gaps::find_cousin_primes(range.clone())) } else { None },
                sexy_primes: if sexy { Some(gaps::find_sexy_primes(range.clone())) } else { None },
                execution_time_ms: start_time.elapsed().as_millis() as u64,
            };
            
            if plot {
                let plot_path = generate_gap_histogram(&output.statistics, "gap_histogram.svg")?;
                println!("Gap histogram saved to: {}", plot_path.display());
            }
            
            print_output(&output, &cli.format, cli.output.as_ref().map(|v| &**v))?;
        }
        
        Commands::Plot { range, output_dir, plot_type } => {
            let range = parse_range(&range)?;
            std::fs::create_dir_all(&output_dir)?;
            
            match plot_type {
                PlotType::All => {
                    generate_all_plots(range, &output_dir)?;
                }
                PlotType::PrimeCount => {
                    let path = generate_prime_count_plot(range, &output_dir.join("prime_count.svg"))?;
                    println!("Prime count plot saved to: {}", path.display());
                }
                PlotType::Gaps => {
                    let stats = gaps::analyze_gaps(range);
                    let path = generate_gap_histogram(&stats, &output_dir.join("gaps.svg"))?;
                    println!("Gap histogram saved to: {}", path.display());
                }
                PlotType::Distribution => {
                    let path = generate_distribution_plot(range, &output_dir.join("distribution.svg"))?;
                    println!("Distribution plot saved to: {}", path.display());
                }
            }
        }
        
        Commands::Benchmark { bench_type, max_n, iterations } => {
            match bench_type {
                BenchmarkType::All => {
                    benchmarks::run_all_benchmarks(max_n, iterations).await?;
                }
                BenchmarkType::Primality => {
                    benchmarks::benchmark_primality(max_n, iterations).await?;
                }
                BenchmarkType::Factorization => {
                    benchmarks::benchmark_factorization(max_n, iterations).await?;
                }
                BenchmarkType::Sieve => {
                    benchmarks::benchmark_sieve(max_n, iterations).await?;
                }
            }
        }
        
        Commands::Web { port, host } => {
            println!("Starting web server at http://{}:{}", host, port);
            // This would start the web server - implemented in prime-tool-web
            println!("Web server functionality available in prime-tool-web crate");
        }
    }
    
    Ok(())
}

/// Parse a range string like "1..1000" or "1..=1000"
fn parse_range(range_str: &str) -> anyhow::Result<Range<u64>> {
    if let Some((start, end)) = range_str.split_once("..") {
        let start: u64 = start.parse()?;
        let end: u64 = if end.starts_with('=') {
            end[1..].parse::<u64>()? + 1
        } else {
            end.parse()?
        };
        Ok(start..end)
    } else {
        Err(anyhow::anyhow!("Invalid range format. Use 'start..end' or 'start..=end'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_range() {
        assert_eq!(parse_range("1..100").unwrap(), 1..100);
        assert_eq!(parse_range("1..=100").unwrap(), 1..101);
        assert_eq!(parse_range("0..1000000").unwrap(), 0..1000000);
    }
    
    #[test]
    fn test_parse_range_invalid() {
        assert!(parse_range("invalid").is_err());
        assert!(parse_range("1.2.3").is_err());
    }
}