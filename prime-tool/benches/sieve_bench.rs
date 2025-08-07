//! Criterion benchmarks for sieve algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prime_tool::*;
use rayon::prelude::*;
use std::ops::Range;

/// Benchmark sieve algorithms by range size
fn bench_sieve_by_range_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("sieve_by_range_size");
    
    let range_configs = vec![
        ("tiny", 2..1_000),
        ("small", 2..10_000),
        ("medium", 2..100_000),
        ("large", 2..1_000_000),
        ("xlarge", 2..10_000_000),
    ];
    
    for (size_name, range) in range_configs {
        let range_size = range.end - range.start;
        group.throughput(Throughput::Elements(range_size));
        
        group.bench_with_input(
            BenchmarkId::new("sequential_sieve", size_name),
            &range,
            |b, range| {
                b.iter(|| {
                    let primes = black_box(sieve_range(black_box(range.clone())));
                    black_box(primes.len());
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel_sieve", size_name),
            &range,
            |b, range| {
                b.iter(|| {
                    let primes = black_box(sieve::parallel_sieve(black_box(range.clone())));
                    black_box(primes.len());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different sieve algorithms
fn bench_sieve_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("sieve_algorithms");
    
    let test_range = 2..1_000_000;
    let range_size = test_range.end - test_range.start;
    group.throughput(Throughput::Elements(range_size));
    
    // Sequential sieve
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let primes = black_box(sieve_range(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Parallel sieve
    group.bench_function("parallel", |b| {
        b.iter(|| {
            let primes = black_box(sieve::parallel_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Segmented sieve
    group.bench_function("segmented", |b| {
        b.iter(|| {
            let primes = black_box(sieve::segmented_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Wheel sieve (if implemented)
    group.bench_function("wheel_sieve", |b| {
        b.iter(|| {
            let primes = black_box(sieve::wheel_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    group.finish();
}

/// Benchmark sieve with different starting points
fn bench_sieve_different_starts(c: &mut Criterion) {
    let mut group = c.benchmark_group("sieve_different_starts");
    
    let range_size = 100_000u64;
    let start_points = vec![
        ("from_2", 2),
        ("from_1000", 1_000),
        ("from_100k", 100_000),
        ("from_1m", 1_000_000),
        ("from_10m", 10_000_000),
    ];
    
    group.throughput(Throughput::Elements(range_size));
    
    for (start_name, start) in start_points {
        let range = start..(start + range_size);
        
        group.bench_with_input(
            BenchmarkId::new("sieve_from_start", start_name),
            &range,
            |b, range| {
                b.iter(|| {
                    let primes = black_box(sieve_range(black_box(range.clone())));
                    black_box(primes.len());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel sieve with different thread counts
fn bench_parallel_sieve_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_sieve_scaling");
    
    let test_range = 2..5_000_000;
    let range_size = test_range.end - test_range.start;
    group.throughput(Throughput::Elements(range_size));
    
    let thread_counts = vec![1, 2, 4, 8, 16];
    
    for thread_count in thread_counts {
        group.bench_with_input(
            BenchmarkId::new("parallel_threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter(|| {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    
                    pool.install(|| {
                        let primes = black_box(sieve::parallel_sieve(black_box(test_range.clone())));
                        black_box(primes.len());
                    });
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark segmented sieve with different segment sizes
fn bench_segmented_sieve_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("segmented_sieve_sizes");
    
    let test_range = 2..1_000_000;
    let range_size = test_range.end - test_range.start;
    group.throughput(Throughput::Elements(range_size));
    
    let segment_sizes = vec![
        ("seg_1k", 1_000),
        ("seg_10k", 10_000),
        ("seg_100k", 100_000),
        ("seg_1m", 1_000_000),
    ];
    
    for (seg_name, segment_size) in segment_sizes {
        group.bench_with_input(
            BenchmarkId::new("segmented_size", seg_name),
            &segment_size,
            |b, &seg_size| {
                b.iter(|| {
                    let primes = black_box(sieve::segmented_sieve_with_size(
                        black_box(test_range.clone()),
                        black_box(seg_size),
                    ));
                    black_box(primes.len());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark sieve memory usage patterns
fn bench_sieve_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("sieve_memory_patterns");
    
    let test_configs = vec![
        ("dense_small", 2..100_000),
        ("dense_large", 2..10_000_000),
        ("sparse_small", 1_000_000..1_100_000),
        ("sparse_large", 100_000_000..101_000_000),
    ];
    
    for (config_name, range) in test_configs {
        let range_size = range.end - range.start;
        group.throughput(Throughput::Elements(range_size));
        
        group.bench_with_input(
            BenchmarkId::new("memory_pattern", config_name),
            &range,
            |b, range| {
                b.iter(|| {
                    let primes = black_box(sieve_range(black_box(range.clone())));
                    black_box(primes.len());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark sieve with prime counting vs full enumeration
fn bench_sieve_counting_vs_enumeration(c: &mut Criterion) {
    let mut group = c.benchmark_group("sieve_counting_vs_enumeration");
    
    let test_range = 2..1_000_000;
    let range_size = test_range.end - test_range.start;
    group.throughput(Throughput::Elements(range_size));
    
    // Full enumeration (return all primes)
    group.bench_function("full_enumeration", |b| {
        b.iter(|| {
            let primes = black_box(sieve_range(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Just counting primes
    group.bench_function("count_only", |b| {
        b.iter(|| {
            let count = black_box(sieve::count_primes_in_range(black_box(test_range.clone())));
            black_box(count);
        });
    });
    
    // Prime checking (using sieve for verification)
    group.bench_function("prime_checking", |b| {
        b.iter(|| {
            let test_numbers = generate_test_numbers_in_range(test_range.clone(), 1000);
            let mut prime_count = 0;
            for n in test_numbers {
                if black_box(is_prime(black_box(n))) {
                    prime_count += 1;
                }
            }
            black_box(prime_count);
        });
    });
    
    group.finish();
}

/// Benchmark sieve performance with different data structures
fn bench_sieve_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("sieve_data_structures");
    
    let test_range = 2..1_000_000;
    let range_size = test_range.end - test_range.start;
    group.throughput(Throughput::Elements(range_size));
    
    // BitVec-based sieve
    group.bench_function("bitvec_sieve", |b| {
        b.iter(|| {
            let primes = black_box(sieve::bitvec_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Vec<bool>-based sieve
    group.bench_function("vec_bool_sieve", |b| {
        b.iter(|| {
            let primes = black_box(sieve::vec_bool_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Compressed sieve (odd numbers only)
    group.bench_function("compressed_sieve", |b| {
        b.iter(|| {
            let primes = black_box(sieve::compressed_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    group.finish();
}

/// Benchmark sieve cache performance
fn bench_sieve_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("sieve_cache_performance");
    
    // Test different cache-friendly patterns
    let cache_configs = vec![
        ("l1_cache", 2..32_000),      // ~32KB, fits in L1
        ("l2_cache", 2..256_000),     // ~256KB, fits in L2
        ("l3_cache", 2..8_000_000),   // ~8MB, fits in L3
        ("memory", 2..100_000_000),   // ~100MB, main memory
    ];
    
    for (cache_name, range) in cache_configs {
        let range_size = range.end - range.start;
        group.throughput(Throughput::Elements(range_size));
        
        group.bench_with_input(
            BenchmarkId::new("cache_level", cache_name),
            &range,
            |b, range| {
                b.iter(|| {
                    let primes = black_box(sieve_range(black_box(range.clone())));
                    black_box(primes.len());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark sieve with different optimization levels
fn bench_sieve_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sieve_optimizations");
    
    let test_range = 2..1_000_000;
    let range_size = test_range.end - test_range.start;
    group.throughput(Throughput::Elements(range_size));
    
    // Basic sieve
    group.bench_function("basic_sieve", |b| {
        b.iter(|| {
            let primes = black_box(sieve::basic_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Optimized sieve (skip even numbers)
    group.bench_function("optimized_sieve", |b| {
        b.iter(|| {
            let primes = black_box(sieve::optimized_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Wheel factorization sieve
    group.bench_function("wheel_factorization", |b| {
        b.iter(|| {
            let primes = black_box(sieve::wheel_factorization_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // SIMD-optimized sieve (if available)
    #[cfg(target_arch = "x86_64")]
    group.bench_function("simd_sieve", |b| {
        b.iter(|| {
            let primes = black_box(sieve::simd_sieve(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    group.finish();
}

/// Benchmark distributed sieve coordination overhead
fn bench_distributed_sieve_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_sieve_overhead");
    
    let test_range = 2..1_000_000;
    let range_size = test_range.end - test_range.start;
    group.throughput(Throughput::Elements(range_size));
    
    // Local sieve (baseline)
    group.bench_function("local_sieve", |b| {
        b.iter(|| {
            let primes = black_box(sieve_range(black_box(test_range.clone())));
            black_box(primes.len());
        });
    });
    
    // Simulated distributed sieve (work unit creation overhead)
    group.bench_function("distributed_simulation", |b| {
        b.iter(|| {
            let work_units = black_box(distributed::create_sieve_work_units(
                black_box(test_range.clone()),
                black_box(4), // 4 workers
            ));
            
            // Simulate processing work units
            let mut total_primes = 0;
            for work_unit in work_units {
                if let distributed::TaskType::PrimeSieve { range } = work_unit.task {
                    let primes = sieve_range(range);
                    total_primes += primes.len();
                }
            }
            
            black_box(total_primes);
        });
    });
    
    group.finish();
}

// Helper functions

/// Generate test numbers in a range for prime checking
fn generate_test_numbers_in_range(range: Range<u64>, count: usize) -> Vec<u64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut numbers = Vec::with_capacity(count);
    
    for _ in 0..count {
        let n = rng.gen_range(range.clone());
        numbers.push(n);
    }
    
    numbers
}

criterion_group!(
    benches,
    bench_sieve_by_range_size,
    bench_sieve_algorithms,
    bench_sieve_different_starts,
    bench_parallel_sieve_scaling,
    bench_segmented_sieve_sizes,
    bench_sieve_memory_patterns,
    bench_sieve_counting_vs_enumeration,
    bench_sieve_data_structures,
    bench_sieve_cache_performance,
    bench_sieve_optimizations,
    bench_distributed_sieve_overhead
);
criterion_main!(benches);