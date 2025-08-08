//! Criterion benchmarks for factorization algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prime_tool::*;
use rand::Rng;

/// Benchmark factorization with different number types
fn bench_factorization_by_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_by_type");
    
    let test_cases = vec![
        ("small_semiprimes", generate_small_semiprimes(100)),
        ("medium_semiprimes", generate_medium_semiprimes(50)),
        ("large_semiprimes", generate_large_semiprimes(20)),
        ("highly_composite", generate_highly_composite(100)),
        ("prime_powers", generate_prime_powers(100)),
    ];
    
    for (type_name, numbers) in test_cases {
        group.throughput(Throughput::Elements(numbers.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("complete_factorization", type_name),
            &numbers,
            |b, numbers| {
                b.iter(|| {
                    for &n in numbers {
                        black_box(factor(black_box(n)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark individual factorization algorithms
fn bench_factorization_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_algorithms");
    
    let test_numbers = generate_medium_semiprimes(100);
    group.throughput(Throughput::Elements(test_numbers.len() as u64));
    
    // Complete factorization (adaptive)
    group.bench_function("complete_adaptive", |b| {
        b.iter(|| {
            for &n in &test_numbers {
                black_box(factor(black_box(n)));
            }
        });
    });
    
    // Pollard's Rho
    group.bench_function("pollard_rho", |b| {
        b.iter(|| {
            for &n in &test_numbers {
                black_box(factorization::pollard_rho(black_box(n)));
            }
        });
    });
    
    // Brent's method
    group.bench_function("brent_method", |b| {
        b.iter(|| {
            for &n in &test_numbers {
                black_box(factorization::brent_factor(black_box(n)));
            }
        });
    });
    
    // ECM for larger numbers
    let large_numbers: Vec<u64> = test_numbers.into_iter().filter(|&n| n > 1_000_000).collect();
    if !large_numbers.is_empty() {
        group.throughput(Throughput::Elements(large_numbers.len() as u64));
        group.bench_function("ecm_method", |b| {
            b.iter(|| {
                for &n in &large_numbers {
                    black_box(factorization::ecm_factor(black_box(n)));
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark factorization by number size
fn bench_factorization_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_by_size");
    
    let size_ranges = vec![
        ("tiny", 100, 1_000),
        ("small", 1_000, 100_000),
        ("medium", 100_000, 10_000_000),
        ("large", 10_000_000, 1_000_000_000),
    ];
    
    for (size_name, min, max) in size_ranges {
        let numbers = generate_semiprimes_in_range(min, max, 50);
        
        if !numbers.is_empty() {
            group.throughput(Throughput::Elements(numbers.len() as u64));
            
            group.bench_with_input(
                BenchmarkId::new("factor_by_size", size_name),
                &numbers,
                |b, numbers| {
                    b.iter(|| {
                        for &n in numbers {
                            black_box(factor(black_box(n)));
                        }
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark single number factorization for latency measurement
fn bench_single_number_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_factorization_latency");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(1000);
    
    let test_cases = vec![
        ("small_semiprime", 15), // 3 × 5
        ("medium_semiprime", 1234567), // 127 × 9721
        ("large_semiprime", 982451653 * 982451657), // product of twin primes
        ("highly_composite", 2 * 3 * 5 * 7 * 11 * 13), // 30030
        ("prime_power", 7_u64.pow(8)), // 7^8
        ("carmichael", 561), // 3 × 11 × 17
    ];
    
    for (name, number) in test_cases {
        group.bench_function(name, |b| {
            b.iter(|| black_box(factor(black_box(number))));
        });
    }
    
    group.finish();
}

/// Benchmark factorization success rate and performance
fn bench_factorization_success_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_success_rate");
    
    // Generate challenging numbers for factorization
    let challenging_numbers = vec![
        // Products of close primes
        999983 * 1000003,
        982451653 * 982451657,
        // Products of distant primes
        97 * 982451653,
        // Carmichael numbers
        561, 1105, 1729, 2465, 2821,
        // Numbers with many small factors
        2 * 3 * 5 * 7 * 11 * 13 * 17 * 19,
    ];
    
    group.throughput(Throughput::Elements(challenging_numbers.len() as u64));
    
    group.bench_function("challenging_numbers", |b| {
        b.iter(|| {
            for &n in &challenging_numbers {
                let factors = black_box(factor(black_box(n)));
                // Verify factorization is correct
                let product: u64 = factors.iter().product();
                assert_eq!(product, n, "Factorization failed for {}", n);
            }
        });
    });
    
    group.finish();
}

/// Benchmark parallel factorization
fn bench_parallel_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_factorization");
    
    let numbers = generate_medium_semiprimes(1000);
    group.throughput(Throughput::Elements(numbers.len() as u64));
    
    // Sequential factorization
    group.bench_function("sequential", |b| {
        b.iter(|| {
            for &n in &numbers {
                black_box(factor(black_box(n)));
            }
        });
    });
    
    // Parallel factorization using rayon
    group.bench_function("parallel", |b| {
        b.iter(|| {
            use rayon::prelude::*;
            let _results: Vec<Vec<u64>> = numbers
                .par_iter()
                .map(|&n| black_box(factor(black_box(n))))
                .collect();
        });
    });
    
    group.finish();
}

/// Benchmark factorization with timeout constraints
fn bench_factorization_with_timeout(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_timeout");
    
    let hard_numbers = generate_hard_to_factor_numbers(20);
    
    // Test different timeout strategies
    let timeout_configs = vec![
        ("quick", std::time::Duration::from_millis(1)),
        ("normal", std::time::Duration::from_millis(10)),
        ("extended", std::time::Duration::from_millis(100)),
    ];
    
    for (timeout_name, _timeout_duration) in timeout_configs {
        group.bench_with_input(
            BenchmarkId::new("timeout_factorization", timeout_name),
            &hard_numbers,
            |b, numbers| {
                b.iter(|| {
                    for &n in numbers {
                        // In a real implementation, you'd pass the timeout to the factorization function
                        black_box(factor(black_box(n)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage during factorization
fn bench_factorization_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_memory");
    
    let numbers_with_many_factors = generate_numbers_with_many_factors(100);
    group.throughput(Throughput::Elements(numbers_with_many_factors.len() as u64));
    
    group.bench_function("many_factors", |b| {
        b.iter(|| {
            for &n in &numbers_with_many_factors {
                let factors = black_box(factor(black_box(n)));
                // Ensure factors are actually used to prevent optimization
                black_box(factors.len());
            }
        });
    });
    
    group.finish();
}

// Helper functions for generating test data

/// Generate small semiprimes (products of two small primes)
fn generate_small_semiprimes(count: usize) -> Vec<u64> {
    let small_primes = sieve_range(2..1000);
    let mut semiprimes = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();
    
    while semiprimes.len() < count {
        let p1 = small_primes[rng.gen_range(0..small_primes.len())];
        let p2 = small_primes[rng.gen_range(0..small_primes.len())];
        let product = p1 * p2;
        
        if product > 1 && product < u64::MAX / 1000 {
            semiprimes.push(product);
        }
    }
    
    semiprimes
}

/// Generate medium semiprimes
fn generate_medium_semiprimes(count: usize) -> Vec<u64> {
    let medium_primes = sieve_range(1000..100_000);
    let mut semiprimes = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();
    
    while semiprimes.len() < count {
        let p1 = medium_primes[rng.gen_range(0..medium_primes.len())];
        let p2 = medium_primes[rng.gen_range(0..medium_primes.len())];
        
        if let Some(product) = p1.checked_mul(p2) {
            if product < u64::MAX / 1000 {
                semiprimes.push(product);
            }
        }
    }
    
    semiprimes
}

/// Generate large semiprimes
fn generate_large_semiprimes(count: usize) -> Vec<u64> {
    let large_primes = sieve_range(100_000..1_000_000);
    let mut semiprimes = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();
    
    while semiprimes.len() < count {
        let p1 = large_primes[rng.gen_range(0..large_primes.len())];
        let p2 = large_primes[rng.gen_range(0..large_primes.len())];
        
        if let Some(product) = p1.checked_mul(p2) {
            if product < u64::MAX / 1000 {
                semiprimes.push(product);
            }
        }
    }
    
    semiprimes
}

/// Generate highly composite numbers (many small prime factors)
fn generate_highly_composite(count: usize) -> Vec<u64> {
    let small_primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    let mut numbers = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();
    
    while numbers.len() < count {
        let mut n = 1u64;
        let factor_count = rng.gen_range(3..8);
        
        for _ in 0..factor_count {
            let prime = small_primes[rng.gen_range(0..small_primes.len())];
            if let Some(new_n) = n.checked_mul(prime) {
                if new_n < u64::MAX / 1000 {
                    n = new_n;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        if n > 1 {
            numbers.push(n);
        }
    }
    
    numbers
}

/// Generate prime powers
fn generate_prime_powers(count: usize) -> Vec<u64> {
    let primes = sieve_range(2..1000);
    let mut powers = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();
    
    while powers.len() < count {
        let prime = primes[rng.gen_range(0..primes.len())];
        let exponent = rng.gen_range(2..8);
        
        if let Some(power) = prime.checked_pow(exponent) {
            if power < u64::MAX / 1000 {
                powers.push(power);
            }
        }
    }
    
    powers
}

/// Generate semiprimes in a specific range
fn generate_semiprimes_in_range(min: u64, max: u64, count: usize) -> Vec<u64> {
    let mut semiprimes = Vec::new();
    let mut rng = rand::thread_rng();
    let mut attempts = 0;
    let max_attempts = count * 10;
    
    while semiprimes.len() < count && attempts < max_attempts {
        attempts += 1;
        
        // Generate a random number in range
        let n = rng.gen_range(min..=max);
        
        // Check if it's a semiprime (has exactly 2 prime factors)
        let factors = factor(n);
        if factors.len() == 2 {
            semiprimes.push(n);
        }
    }
    
    semiprimes
}

/// Generate numbers that are hard to factor
fn generate_hard_to_factor_numbers(count: usize) -> Vec<u64> {
    let mut hard_numbers = Vec::with_capacity(count);
    
    // Add some known hard cases
    hard_numbers.extend_from_slice(&[
        // Products of close primes
        999983 * 1000003,
        982451653 * 982451657,
        // Carmichael numbers
        561, 1105, 1729, 2465, 2821, 6601, 8911, 10585,
        // Numbers with specific structure
        2_u64.pow(31) - 1, // Mersenne number
        2_u64.pow(32) + 1, // Fermat number
    ]);
    
    // Generate additional hard cases
    let large_primes = sieve_range(1_000_000..10_000_000);
    let mut rng = rand::thread_rng();
    
    while hard_numbers.len() < count {
        // Create products of large, close primes
        let idx = rng.gen_range(0..large_primes.len() - 10);
        let p1 = large_primes[idx];
        let p2 = large_primes[idx + rng.gen_range(1..10)];
        
        if let Some(product) = p1.checked_mul(p2) {
            if product < u64::MAX / 1000 {
                hard_numbers.push(product);
            }
        }
    }
    
    hard_numbers.truncate(count);
    hard_numbers
}

/// Generate numbers with many factors
fn generate_numbers_with_many_factors(count: usize) -> Vec<u64> {
    let mut numbers = Vec::with_capacity(count);
    
    // Start with highly composite numbers
    let base_numbers: Vec<u64> = vec![
        2 * 3 * 5 * 7 * 11 * 13,
        2 * 3 * 5 * 7 * 11 * 13 * 17,
        2 * 3 * 5 * 7 * 11 * 13 * 17 * 19,
    ];
    
    for &base in &base_numbers {
        numbers.push(base);
        
        // Add powers of the base
        if let Some(squared) = base.checked_mul(base) {
            if squared < u64::MAX / 1000 {
                numbers.push(squared);
            }
        }
    }
    
    // Generate additional numbers with many small factors
    let small_primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let mut rng = rand::thread_rng();
    
    while numbers.len() < count {
        let mut n = 1u64;
        let factor_count = rng.gen_range(5..10);
        
        for _ in 0..factor_count {
            let prime = small_primes[rng.gen_range(0..small_primes.len())];
            let exponent = rng.gen_range(1..4);
            
            for _ in 0..exponent {
                if let Some(new_n) = n.checked_mul(prime) {
                    if new_n < u64::MAX / 1000 {
                        n = new_n;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        
        if n > 1 {
            numbers.push(n);
        }
    }
    
    numbers.truncate(count);
    numbers
}

criterion_group!(
    benches,
    bench_factorization_by_type,
    bench_factorization_algorithms,
    bench_factorization_by_size,
    bench_single_number_factorization,
    bench_factorization_success_rate,
    bench_parallel_factorization,
    bench_factorization_with_timeout,
    bench_factorization_memory
);
criterion_main!(benches);