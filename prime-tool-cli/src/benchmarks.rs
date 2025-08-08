//! Benchmarking module for performance testing

use prime_tool::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use comfy_table::{Table, Cell, Attribute, Color};
use rand::Rng;

/// Run all available benchmarks
pub async fn run_all_benchmarks(max_n: u64, iterations: usize) -> anyhow::Result<()> {
    println!("Running comprehensive benchmarks...");
    println!("Max N: {}, Iterations: {}", max_n, iterations);
    println!("{}", "=".repeat(60));
    
    benchmark_primality(max_n, iterations).await?;
    println!();
    benchmark_factorization(max_n, iterations).await?;
    println!();
    benchmark_sieve(max_n, iterations).await?;
    println!();
    benchmark_gap_analysis(max_n, iterations).await?;
    
    Ok(())
}

/// Benchmark primality testing algorithms
pub async fn benchmark_primality(max_n: u64, iterations: usize) -> anyhow::Result<()> {
    println!("🔍 Primality Testing Benchmarks");
    println!("{}", "-".repeat(40));
    
    let test_numbers = generate_test_numbers(max_n, iterations);
    
    // Benchmark different algorithms
    let mut results = HashMap::new();
    
    // Test is_prime (adaptive algorithm)
    let start = Instant::now();
    let mut correct_results = 0;
    for &n in &test_numbers {
        if is_prime(n) {
            correct_results += 1;
        }
    }
    let adaptive_time = start.elapsed();
    results.insert("Adaptive (Miller-Rabin + Baillie-PSW)".to_string(), adaptive_time);
    
    // Test Miller-Rabin with different witness sets
    let start = Instant::now();
    for &n in &test_numbers {
        algorithms::miller_rabin(n, &[2, 3, 5, 7, 11, 13, 17]);
    }
    let miller_rabin_time = start.elapsed();
    results.insert("Miller-Rabin (7 witnesses)".to_string(), miller_rabin_time);
    
    // Test Baillie-PSW
    let start = Instant::now();
    for &n in &test_numbers {
        algorithms::baillie_psw(n);
    }
    let baillie_psw_time = start.elapsed();
    results.insert("Baillie-PSW".to_string(), baillie_psw_time);
    
    // Display results
    display_benchmark_results(&results, iterations, "Primality Tests");
    
    // Performance analysis
    let avg_time_per_test = adaptive_time.as_nanos() as f64 / iterations as f64;
    let tests_per_second = 1_000_000_000.0 / avg_time_per_test;
    
    println!("\n📊 Performance Analysis:");
    println!("Average time per test: {:.2} ns", avg_time_per_test);
    println!("Tests per second: {:.0}", tests_per_second);
    println!("Correct prime identifications: {}", correct_results);
    
    // Check if we meet the sub-microsecond requirement
    if avg_time_per_test < 1000.0 {
        println!("✅ Sub-microsecond requirement MET!");
    } else {
        println!("❌ Sub-microsecond requirement NOT met");
    }
    
    Ok(())
}

/// Benchmark factorization algorithms
pub async fn benchmark_factorization(max_n: u64, iterations: usize) -> anyhow::Result<()> {
    println!("🔧 Factorization Benchmarks");
    println!("{}", "-".repeat(40));
    
    let test_numbers = generate_composite_numbers(max_n, iterations);
    
    let mut results = HashMap::new();
    
    // Test complete factorization
    let start = Instant::now();
    for &n in &test_numbers {
        factor(n);
    }
    let complete_time = start.elapsed();
    results.insert("Complete Factorization".to_string(), complete_time);
    
    // Test Pollard's Rho
    let start = Instant::now();
    for &n in &test_numbers {
        factorization::pollard_rho(n);
    }
    let pollard_time = start.elapsed();
    results.insert("Pollard's Rho".to_string(), pollard_time);
    
    // Test Brent's method
    let start = Instant::now();
    for &n in &test_numbers {
        factorization::brent_factor(n);
    }
    let brent_time = start.elapsed();
    results.insert("Brent's Method".to_string(), brent_time);
    
    // Test ECM for larger numbers
    let large_numbers: Vec<u64> = test_numbers.iter().filter(|&&n| n > 1_000_000).cloned().collect();
    if !large_numbers.is_empty() {
        let start = Instant::now();
        for &n in &large_numbers {
            factorization::ecm_factor(n);
        }
        let ecm_time = start.elapsed();
        results.insert("Elliptic Curve Method".to_string(), ecm_time);
    }
    
    display_benchmark_results(&results, iterations, "Factorization Methods");
    
    Ok(())
}

/// Benchmark sieving algorithms
pub async fn benchmark_sieve(max_n: u64, _iterations: usize) -> anyhow::Result<()> {
    println!("🔢 Sieve Benchmarks");
    println!("{}", "-".repeat(40));
    
    let mut results = HashMap::new();
    
    // Test different sieve ranges
    let ranges = vec![
        1..1_000,
        1..10_000,
        1..100_000,
        1..max_n.min(1_000_000),
    ];
    
    for range in ranges {
        let range_size = range.end - range.start;
        
        // Sequential sieve
        let start = Instant::now();
        let primes = sieve_range(range.clone());
        let sequential_time = start.elapsed();
        
        // Parallel sieve
        let start = Instant::now();
        let parallel_primes = sieve_parallel(range.clone(), 10_000);
        let parallel_time = start.elapsed();
        
        // Verify results match
        assert_eq!(primes.len(), parallel_primes.len());
        
        results.insert(
            format!("Sequential Sieve ({})", format_number(range_size)),
            sequential_time,
        );
        results.insert(
            format!("Parallel Sieve ({})", format_number(range_size)),
            parallel_time,
        );
        
        println!("Range 1..{}: {} primes found", format_number(range.end), primes.len());
        
        // Calculate primes per second
        let primes_per_sec = primes.len() as f64 / sequential_time.as_secs_f64();
        println!("  Sequential: {:.0} primes/sec", primes_per_sec);
        
        let parallel_primes_per_sec = parallel_primes.len() as f64 / parallel_time.as_secs_f64();
        println!("  Parallel: {:.0} primes/sec", parallel_primes_per_sec);
        
        let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("  Speedup: {:.2}x", speedup);
        println!();
    }
    
    display_benchmark_results(&results, 1, "Sieve Performance");
    
    Ok(())
}

/// Benchmark gap analysis
pub async fn benchmark_gap_analysis(max_n: u64, _iterations: usize) -> anyhow::Result<()> {
    println!("📏 Gap Analysis Benchmarks");
    println!("{}", "-".repeat(40));
    
    let ranges = vec![
        1..10_000,
        1..100_000,
        1..max_n.min(1_000_000),
    ];
    
    let mut results = HashMap::new();
    
    for range in ranges {
        let range_size = range.end - range.start;
        
        // Gap analysis
        let start = Instant::now();
        let stats = gaps::analyze_gaps(range.clone());
        let gap_time = start.elapsed();
        
        // Twin prime finding
        let start = Instant::now();
        let twins = gaps::find_twin_primes(range.clone());
        let twin_time = start.elapsed();
        
        results.insert(
            format!("Gap Analysis ({})", format_number(range_size)),
            gap_time,
        );
        results.insert(
            format!("Twin Prime Search ({})", format_number(range_size)),
            twin_time,
        );
        
        println!("Range 1..{}:", format_number(range.end));
        println!("  Total gaps: {}", stats.total_gaps);
        println!("  Max gap: {}", stats.max_gap);
        println!("  Average gap: {:.2}", stats.average_gap);
        println!("  Twin primes: {}", twins.len());
        println!();
    }
    
    display_benchmark_results(&results, 1, "Gap Analysis Performance");
    
    Ok(())
}

/// Generate test numbers for benchmarking
fn generate_test_numbers(max_n: u64, count: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let mut numbers = Vec::with_capacity(count);
    
    // Include some known primes and composites
    let known_primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 97, 101, 103, 107, 109, 113];
    let known_composites = vec![4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28];
    
    // Add known numbers
    for &p in &known_primes {
        if p <= max_n && numbers.len() < count {
            numbers.push(p);
        }
    }
    
    for &c in &known_composites {
        if c <= max_n && numbers.len() < count {
            numbers.push(c);
        }
    }
    
    // Fill with random numbers
    while numbers.len() < count {
        let n = rng.gen_range(2..=max_n);
        numbers.push(n);
    }
    
    numbers.truncate(count);
    numbers
}

/// Generate composite numbers for factorization testing
fn generate_composite_numbers(max_n: u64, count: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let mut numbers = Vec::with_capacity(count);
    
    // Generate products of small primes
    let small_primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    
    while numbers.len() < count {
        let p1 = small_primes[rng.gen_range(0..small_primes.len())];
        let p2 = small_primes[rng.gen_range(0..small_primes.len())];
        let product = p1 * p2;
        
        if product <= max_n && product > 1 {
            numbers.push(product);
        }
        
        // Also add some larger random composites
        if numbers.len() < count / 2 {
            let n = rng.gen_range(100..=max_n);
            if !is_prime(n) {
                numbers.push(n);
            }
        }
    }
    
    numbers.truncate(count);
    numbers
}

/// Display benchmark results in a formatted table
fn display_benchmark_results(
    results: &HashMap<String, Duration>,
    iterations: usize,
    title: &str,
) {
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Algorithm").add_attribute(Attribute::Bold),
        Cell::new("Total Time").add_attribute(Attribute::Bold),
        Cell::new("Avg per Op").add_attribute(Attribute::Bold),
        Cell::new("Ops/sec").add_attribute(Attribute::Bold),
    ]);
    
    let mut sorted_results: Vec<_> = results.iter().collect();
    sorted_results.sort_by_key(|(_, duration)| *duration);
    
    for (algorithm, duration) in sorted_results {
        let total_ms = duration.as_millis();
        let avg_ns = duration.as_nanos() as f64 / iterations as f64;
        let ops_per_sec = 1_000_000_000.0 / avg_ns;
        
        let avg_cell = if avg_ns < 1000.0 {
            Cell::new(&format!("{:.0} ns", avg_ns)).fg(Color::Green)
        } else if avg_ns < 1_000_000.0 {
            Cell::new(&format!("{:.2} μs", avg_ns / 1000.0)).fg(Color::Yellow)
        } else {
            Cell::new(&format!("{:.2} ms", avg_ns / 1_000_000.0)).fg(Color::Red)
        };
        
        table.add_row(vec![
            Cell::new(algorithm),
            Cell::new(&format!("{} ms", total_ms)),
            avg_cell,
            Cell::new(&format!("{:.0}", ops_per_sec)),
        ]);
    }
    
    println!("\n{}", title);
    println!("{}", table);
}

/// Format large numbers with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_test_numbers() {
        let numbers = generate_test_numbers(100, 50);
        assert_eq!(numbers.len(), 50);
        assert!(numbers.iter().all(|&n| n >= 2 && n <= 100));
    }
    
    #[test]
    fn test_generate_composite_numbers() {
        let numbers = generate_composite_numbers(1000, 20);
        assert_eq!(numbers.len(), 20);
        // Most should be composite (allowing for some randomness)
        let composite_count = numbers.iter().filter(|&&n| !is_prime(n)).count();
        assert!(composite_count >= numbers.len() / 2);
    }
    
    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234), "1,234");
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(format_number(123), "123");
    }
    
    #[tokio::test]
    async fn test_benchmark_primality() {
        // Quick test with small numbers
        let result = benchmark_primality(1000, 10).await;
        assert!(result.is_ok());
    }
}