//! Criterion benchmarks for primality testing algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prime_tool::*;
use rand::Rng;

/// Benchmark primality testing with different number sizes
fn bench_primality_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("primality_by_size");
    
    // Test different number ranges
    let test_cases = vec![
        ("small", generate_test_numbers(100, 1000)),
        ("medium", generate_test_numbers(1_000_000, 100_000_000)),
        ("large", generate_test_numbers(1_000_000_000, u64::MAX / 1000)),
    ];
    
    for (size_name, numbers) in test_cases {
        group.throughput(Throughput::Elements(numbers.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("is_prime", size_name),
            &numbers,
            |b, numbers| {
                b.iter(|| {
                    for &n in numbers {
                        black_box(is_prime(black_box(n)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different primality testing algorithms
fn bench_primality_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("primality_algorithms");
    
    let test_numbers = generate_test_numbers(1000, 1_000_000);
    group.throughput(Throughput::Elements(test_numbers.len() as u64));
    
    // Adaptive algorithm (default)
    group.bench_function("adaptive", |b| {
        b.iter(|| {
            for &n in &test_numbers {
                black_box(is_prime(black_box(n)));
            }
        });
    });
    
    // Miller-Rabin with standard witnesses
    group.bench_function("miller_rabin_standard", |b| {
        b.iter(|| {
            for &n in &test_numbers {
                black_box(algorithms::miller_rabin(black_box(n), &[2, 3, 5, 7, 11, 13, 17]));
            }
        });
    });
    
    // Miller-Rabin with extended witnesses
    group.bench_function("miller_rabin_extended", |b| {
        b.iter(|| {
            for &n in &test_numbers {
                black_box(algorithms::miller_rabin(
                    black_box(n),
                    &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
                ));
            }
        });
    });
    
    // Baillie-PSW for larger numbers
    let large_numbers: Vec<u64> = test_numbers.into_iter().filter(|&n| n > 1_000_000).collect();
    if !large_numbers.is_empty() {
        group.throughput(Throughput::Elements(large_numbers.len() as u64));
        group.bench_function("baillie_psw", |b| {
            b.iter(|| {
                for &n in &large_numbers {
                    black_box(algorithms::baillie_psw(black_box(n)));
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark primality testing with known primes vs composites
fn bench_primality_by_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("primality_by_type");
    
    // Generate known primes and composites
    let primes = sieve_range(1000..100_000);
    let primes_sample: Vec<u64> = primes.into_iter().take(1000).collect();
    
    let composites = generate_composites(1000);
    
    group.throughput(Throughput::Elements(primes_sample.len() as u64));
    
    group.bench_function("known_primes", |b| {
        b.iter(|| {
            for &n in &primes_sample {
                black_box(is_prime(black_box(n)));
            }
        });
    });
    
    group.throughput(Throughput::Elements(composites.len() as u64));
    
    group.bench_function("known_composites", |b| {
        b.iter(|| {
            for &n in &composites {
                black_box(is_prime(black_box(n)));
            }
        });
    });
    
    group.finish();
}

/// Benchmark primality testing with Carmichael numbers (pseudoprimes)
fn bench_carmichael_numbers(c: &mut Criterion) {
    let mut group = c.benchmark_group("carmichael_numbers");
    
    // Known Carmichael numbers
    let carmichael_numbers = vec![
        561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341,
        41041, 46657, 52633, 62745, 63973, 75361, 101101, 115921, 126217, 162401
    ];
    
    group.throughput(Throughput::Elements(carmichael_numbers.len() as u64));
    
    group.bench_function("carmichael_adaptive", |b| {
        b.iter(|| {
            for &n in &carmichael_numbers {
                black_box(is_prime(black_box(n)));
            }
        });
    });
    
    group.bench_function("carmichael_miller_rabin", |b| {
        b.iter(|| {
            for &n in &carmichael_numbers {
                black_box(algorithms::miller_rabin(black_box(n), &[2, 3, 5, 7, 11, 13, 17]));
            }
        });
    });
    
    group.bench_function("carmichael_baillie_psw", |b| {
        b.iter(|| {
            for &n in &carmichael_numbers {
                black_box(algorithms::baillie_psw(black_box(n)));
            }
        });
    });
    
    group.finish();
}

/// Benchmark single number primality testing for latency measurement
fn bench_single_number_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_number_latency");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(10000);
    
    let test_cases = vec![
        ("small_prime", 997),
        ("medium_prime", 999983),
        ("large_prime", 982451653),
        ("small_composite", 1001),
        ("medium_composite", 999984),
        ("large_composite", 982451654),
    ];
    
    for (name, number) in test_cases {
        group.bench_function(name, |b| {
            b.iter(|| black_box(is_prime(black_box(number))));
        });
    }
    
    group.finish();
}

/// Benchmark performance scaling with input size
fn bench_scaling_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_performance");
    
    let sizes = vec![10, 100, 1000, 10000];
    
    for size in sizes {
        let numbers = generate_test_numbers(1000, 100_000);
        let sample: Vec<u64> = numbers.into_iter().take(size).collect();
        
        group.throughput(Throughput::Elements(sample.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_primality", size),
            &sample,
            |b, sample| {
                b.iter(|| {
                    for &n in sample {
                        black_box(is_prime(black_box(n)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Generate test numbers in a given range
fn generate_test_numbers(min: u64, max: u64) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let mut numbers = Vec::with_capacity(1000);
    
    for _ in 0..1000 {
        numbers.push(rng.gen_range(min..=max));
    }
    
    numbers
}

/// Generate known composite numbers
fn generate_composites(count: usize) -> Vec<u64> {
    let mut composites = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();
    
    while composites.len() < count {
        let n = rng.gen_range(4..1_000_000);
        if !is_prime(n) {
            composites.push(n);
        }
    }
    
    composites
}

criterion_group!(
    benches,
    bench_primality_by_size,
    bench_primality_algorithms,
    bench_primality_by_type,
    bench_carmichael_numbers,
    bench_single_number_latency,
    bench_scaling_performance
);
criterion_main!(benches);