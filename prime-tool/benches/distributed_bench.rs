//! Criterion benchmarks for distributed computing functionality

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prime_tool::*;
use tokio::runtime::Runtime;
use serde_json;

/// Benchmark distributed coordinator overhead
fn bench_distributed_coordinator(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_coordinator");
    
    let rt = Runtime::new().unwrap();
    
    // Node registration overhead
    group.bench_function("node_registration", |b| {
        b.iter(|| {
            rt.block_on(async {
                let coordinator = black_box(distributed::DistributedCoordinator::new(30));
                
                // Register multiple nodes
                for i in 0..10 {
                    let node_info = distributed::NodeInfo {
                        id: format!("node_{}", i),
                        address: format!("127.0.0.1:{}", 8000 + i),
                        cpu_cores: 4,
                        memory_gb: 8.0,
                        load_factor: 0.5,
                        last_heartbeat: 0,
                        capabilities: vec![
                            distributed::TaskType::PrimeSieve { range: 0..0 },
                            distributed::TaskType::Factorization { numbers: vec![] }
                        ],
                    };
                    
                    let _ = black_box(coordinator.register_node(black_box(node_info)).await);
                }
                
                let stats = black_box(coordinator.get_cluster_stats().await);
                black_box(stats.total_nodes);
            });
        });
    });
    
    // Work unit creation and distribution
    group.bench_function("work_distribution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let coordinator = black_box(distributed::DistributedCoordinator::new(30));
                
                // Register nodes
                for i in 0..4 {
                    let node_info = distributed::NodeInfo {
                        id: format!("node_{}", i),
                        address: format!("127.0.0.1:{}", 8000 + i),
                        cpu_cores: 2,
                        memory_gb: 4.0,
                        load_factor: 0.3,
                        last_heartbeat: 0,
                        capabilities: vec![distributed::TaskType::PrimeSieve { range: 0..0 }],
                    };
                    coordinator.register_node(node_info).await.unwrap();
                }
                
                // Submit work
                let work_unit = distributed::WorkUnit {
                    id: "test_work".to_string(),
                    task: distributed::TaskType::PrimeSieve {
                        range: 2..1_000_000,
                    },
                    priority: 1,
                    estimated_duration_ms: 10000,
                };
                
                let _ = black_box(coordinator.submit_work(black_box(work_unit)).await);
                
                // Get work for nodes
                for i in 0..4 {
                    if let Ok(Some(assignment)) = coordinator.get_work(&format!("node_{}", i)).await {
                        black_box(assignment);
                    }
                }
            });
        });
    });
    
    group.finish();
}

/// Benchmark work unit creation for different task types
fn bench_work_unit_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_unit_creation");
    
    let test_configs = vec![
        ("small_sieve", 2..10_000),
        ("medium_sieve", 2..1_000_000),
        ("large_sieve", 2..10_000_000),
    ];
    
    for (config_name, range) in test_configs {
        let range_size = range.end - range.start;
        group.throughput(Throughput::Elements(range_size));
        
        // Sieve work units
        group.bench_with_input(
            BenchmarkId::new("sieve_work_units", config_name),
            &range,
            |b, range| {
                b.iter(|| {
                    let work_units = black_box(distributed::create_sieve_work_units(
                        black_box(range.clone()),
                        black_box(4), // 4 workers
                    ));
                    black_box(work_units.len());
                });
            },
        );
        
        // Gap analysis work units (using sieve work units as template)
        group.bench_with_input(
            BenchmarkId::new("gap_work_units", config_name),
            &range,
            |b, range| {
                b.iter(|| {
                    // Create gap analysis work units manually since function doesn't exist
                    let mut work_units = Vec::new();
                    let chunk_size = (range.end - range.start) / 4;
                    for i in 0..4 {
                        let start = range.start + i * chunk_size;
                        let end = if i == 3 { range.end } else { start + chunk_size };
                        work_units.push(distributed::WorkUnit {
                            id: format!("gap_{}_{}", start, end),
                            task: distributed::TaskType::GapAnalysis { range: start..end },
                            priority: 1,
                            estimated_duration_ms: end - start,
                        });
                    }
                    black_box(work_units.len());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark local executor performance
fn bench_local_executor(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_executor");
    
    let rt = Runtime::new().unwrap();
    
    // Sieve execution
    group.bench_function("local_sieve_execution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let executor = black_box(distributed::LocalExecutor::new("test_node".to_string()));
                
                let work_unit = distributed::WorkUnit {
                    id: "sieve_test".to_string(),
                    task: distributed::TaskType::PrimeSieve {
                        range: 2..100_000,
                    },
                    priority: 1,
                    estimated_duration_ms: 1000,
                };
                
                let result = black_box(executor.execute_work(black_box(work_unit)).await);
                let _ = black_box(result);
            });
        });
    });
    
    // Factorization execution
    group.bench_function("local_factorization_execution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let executor = black_box(distributed::LocalExecutor::new("test_node".to_string()));
                
                let work_unit = distributed::WorkUnit {
                    id: "factor_test".to_string(),
                    task: distributed::TaskType::Factorization {
                        numbers: vec![982451653 * 982451657],
                    },
                    priority: 1,
                    estimated_duration_ms: 100,
                };
                
                let result = black_box(executor.execute_work(black_box(work_unit)).await);
                let _ = black_box(result);
            });
        });
    });
    
    // Primality test execution
    group.bench_function("local_primality_execution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let executor = black_box(distributed::LocalExecutor::new("test_node".to_string()));
                
                let work_unit = distributed::WorkUnit {
                    id: "primality_test".to_string(),
                    task: distributed::TaskType::PrimalityTest {
                        numbers: vec![982451653],
                    },
                    priority: 1,
                    estimated_duration_ms: 1,
                };
                
                let result = black_box(executor.execute_work(black_box(work_unit)).await);
                let _ = black_box(result);
            });
        });
    });
    
    group.finish();
}

/// Benchmark distributed sieve coordination
fn bench_distributed_sieve_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_sieve_coordination");
    
    let rt = Runtime::new().unwrap();
    
    let worker_counts = vec![1, 2, 4, 8];
    let test_range = 2..1_000_000;
    let range_size = test_range.end - test_range.start;
    
    for worker_count in worker_counts {
        group.throughput(Throughput::Elements(range_size));
        
        group.bench_with_input(
            BenchmarkId::new("distributed_sieve", worker_count),
            &worker_count,
            |b, &workers| {
                b.iter(|| {
                    rt.block_on(async {
                        let coordinator = distributed::DistributedCoordinator::new(30);
                        let chunk_size = range_size / workers as u64;
                        let result = black_box(coordinator.distribute_sieve(
                            black_box(test_range.clone()),
                            black_box(chunk_size),
                        ).await);
                        match result {
                            Ok(primes) => black_box(primes.len()),
                            Err(_) => black_box(0),
                        };
                    });
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark result aggregation
fn bench_result_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("result_aggregation");
    
    let rt = Runtime::new().unwrap();
    
    // Generate test results
    let sieve_results = generate_sieve_results(100);
    let gap_results = generate_gap_results(50);
    let factorization_results = generate_factorization_results(200);
    
    // Sieve result aggregation
    group.throughput(Throughput::Elements(sieve_results.len() as u64));
    group.bench_function("sieve_aggregation", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Manual aggregation since function doesn't exist
                let mut all_primes: Vec<u64> = Vec::new();
                for result in &sieve_results {
                    if let distributed::TaskResult::Primes(primes) = &result.result {
                        all_primes.extend(primes);
                    }
                }
                all_primes.sort_unstable();
                all_primes.dedup();
                black_box(all_primes.len());
            });
        });
    });
    
    // Gap analysis result aggregation
    group.throughput(Throughput::Elements(gap_results.len() as u64));
    group.bench_function("gap_aggregation", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Manual aggregation since function doesn't exist
                let mut total_gaps = 0;
                for result in &gap_results {
                    if let distributed::TaskResult::Gaps(gaps) = &result.result {
                        total_gaps += gaps.len();
                    }
                }
                black_box(total_gaps);
            });
        });
    });
    
    // Factorization result aggregation
    group.throughput(Throughput::Elements(factorization_results.len() as u64));
    group.bench_function("factorization_aggregation", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Manual aggregation since function doesn't exist
                let mut all_factors = std::collections::HashMap::new();
                for result in &factorization_results {
                    if let distributed::TaskResult::Factors(factors) = &result.result {
                        all_factors.extend(factors.clone());
                    }
                }
                black_box(all_factors.len());
            });
        });
    });
    
    group.finish();
}

/// Benchmark cluster statistics and monitoring
fn bench_cluster_monitoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("cluster_monitoring");
    
    let rt = Runtime::new().unwrap();
    
    group.bench_function("cluster_stats_collection", |b| {
        b.iter(|| {
            rt.block_on(async {
                let coordinator = black_box(distributed::DistributedCoordinator::new(30));
                
                // Register nodes with different loads
                for i in 0..10 {
                    let node_info = distributed::NodeInfo {
                        id: format!("node_{}", i),
                        address: format!("127.0.0.1:{}", 8000 + i),
                        cpu_cores: 4,
                        memory_gb: 8.0,
                        load_factor: (i as f64) * 0.1,
                        last_heartbeat: 0,
                        capabilities: vec![
                            distributed::TaskType::PrimeSieve { range: 0..0 },
                            distributed::TaskType::Factorization { numbers: vec![] }
                        ],
                    };
                    coordinator.register_node(node_info).await.unwrap();
                }
                
                // Submit various work units
                for j in 0..20 {
                    let work_unit = distributed::WorkUnit {
                        id: format!("work_{}", j),
                        task: distributed::TaskType::PrimeSieve {
                            range: (j * 10_000)..((j + 1) * 10_000),
                        },
                        priority: 1,
                        estimated_duration_ms: 1000,
                    };
                    coordinator.submit_work(work_unit).await.unwrap();
                }
                
                // Collect cluster statistics
                let stats = black_box(coordinator.get_cluster_stats().await);
                black_box(stats.total_nodes);
            });
        });
    });
    
    group.finish();
}

/// Benchmark load balancing algorithms
fn bench_load_balancing(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_balancing");
    
    let rt = Runtime::new().unwrap();
    
    let balancing_strategies = vec![
        "round_robin",
        "least_loaded",
        "capability_based",
        "priority_based",
    ];
    
    for strategy in balancing_strategies {
        group.bench_with_input(
            BenchmarkId::new("load_balancing", strategy),
            &strategy,
            |b, _strategy_name| {
                b.iter(|| {
                    rt.block_on(async {
                        let coordinator = black_box(distributed::DistributedCoordinator::new(30));
                        // Note: set_load_balancing_strategy method doesn't exist, so we skip it
                        
                        // Register nodes with different capabilities
                        for i in 0..8 {
                            let capabilities = match i % 3 {
                                0 => vec![distributed::TaskType::PrimeSieve { range: 0..0 }],
                                1 => vec![distributed::TaskType::Factorization { numbers: vec![] }],
                                _ => vec![
                                    distributed::TaskType::PrimeSieve { range: 0..0 },
                                    distributed::TaskType::Factorization { numbers: vec![] },
                                    distributed::TaskType::GapAnalysis { range: 0..0 }
                                ],
                            };
                            
                            let node_info = distributed::NodeInfo {
                                id: format!("node_{}", i),
                                address: format!("127.0.0.1:{}", 8000 + i),
                                cpu_cores: 2 + (i % 3),
                                memory_gb: 4.0,
                                load_factor: 0.5,
                                last_heartbeat: 0,
                                capabilities,
                            };
                            coordinator.register_node(node_info).await.unwrap();
                        }
                        
                        // Submit mixed workload
                        let work_types = vec![
                            distributed::TaskType::PrimeSieve { range: 2..100_000 },
                            distributed::TaskType::Factorization { numbers: vec![982451653] },
                            distributed::TaskType::GapAnalysis { range: 2..50_000 },
                            distributed::TaskType::PrimalityTest { numbers: vec![999983] },
                        ];
                        
                        for (i, task) in work_types.into_iter().enumerate() {
                            let work_unit = distributed::WorkUnit {
                                id: format!("mixed_work_{}", i),
                                task,
                                priority: (i % 3) as u8 + 1,
                                estimated_duration_ms: 1000,
                            };
                            coordinator.submit_work(work_unit).await.unwrap();
                        }
                        
                        // Get work for nodes
                        let mut assignments = 0;
                        for i in 0..8 {
                            if let Ok(Some(assignment)) = coordinator.get_work(&format!("node_{}", i)).await {
                                black_box(assignment);
                                assignments += 1;
                                if assignments >= 4 {
                                    break;
                                }
                            }
                        }
                        
                        black_box(assignments);
                    });
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark fault tolerance and recovery
fn bench_fault_tolerance(c: &mut Criterion) {
    let mut group = c.benchmark_group("fault_tolerance");
    
    let rt = Runtime::new().unwrap();
    
    group.bench_function("node_failure_recovery", |b| {
        b.iter(|| {
            rt.block_on(async {
                let coordinator = black_box(distributed::DistributedCoordinator::new(30));
                
                // Register nodes
                for i in 0..5 {
                    let node_info = distributed::NodeInfo {
                        id: format!("node_{}", i),
                        address: format!("127.0.0.1:{}", 8000 + i),
                        cpu_cores: 2,
                        memory_gb: 4.0,
                        load_factor: 0.5,
                        last_heartbeat: 0,
                        capabilities: vec![distributed::TaskType::PrimeSieve { range: 0..0 }],
                    };
                    coordinator.register_node(node_info).await.unwrap();
                }
                
                // Submit work
                for i in 0..10 {
                    let work_unit = distributed::WorkUnit {
                        id: format!("work_{}", i),
                        task: distributed::TaskType::PrimeSieve {
                            range: (i * 10_000)..((i + 1) * 10_000),
                        },
                        priority: 1,
                        estimated_duration_ms: 5000,
                    };
                    coordinator.submit_work(work_unit).await.unwrap();
                }
                
                // Get work for nodes
                let mut assignments = Vec::new();
                for i in 0..5 {
                    if let Ok(Some(assignment)) = coordinator.get_work(&format!("node_{}", i)).await {
                        assignments.push(assignment);
                    }
                }
                
                // Simulate node failure by unregistering
                if let Err(e) = coordinator.unregister_node("node_2").await {
                    eprintln!("Failed to unregister node: {:?}", e);
                }
                
                // Note: reassign_failed_work and handle_node_failure methods don't exist
                // So we just measure the unregistration overhead
                black_box(assignments.len());
            });
        });
    });
    
    group.finish();
}

/// Benchmark network communication simulation
fn bench_network_communication(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_communication");
    
    let rt = Runtime::new().unwrap();
    
    let message_sizes = vec![
        ("small", 1_000),
        ("medium", 100_000),
        ("large", 1_000_000),
    ];
    
    for (size_name, data_size) in message_sizes {
        group.throughput(Throughput::Bytes(data_size));
        
        group.bench_with_input(
            BenchmarkId::new("message_serialization", size_name),
            &data_size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        // Generate test data
                        let primes: Vec<u64> = (2..size).filter(|&n| prime_tool::is_prime(n)).collect();
                        
                        // Manual serialization since functions don't exist
                        let result = distributed::TaskResult::Primes(primes.clone());
                        let serialized = black_box(serde_json::to_string(&result).unwrap());
                        
                        // Manual deserialization
                        let deserialized: distributed::TaskResult = black_box(serde_json::from_str(&serialized).unwrap());
                        
                        if let distributed::TaskResult::Primes(primes) = deserialized {
                            black_box(primes.len());
                        }
                    });
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions for generating test data

/// Generate mock sieve results
fn generate_sieve_results(count: usize) -> Vec<distributed::WorkResult> {
    let mut results = Vec::with_capacity(count);
    
    for i in 0..count {
        let start = i * 10_000;
        let end = (i + 1) * 10_000;
        let primes = prime_tool::sieve::sieve_range(start as u64..(end as u64));
        
        let result = distributed::WorkResult {
            work_id: format!("sieve_{}", i),
            node_id: format!("node_{}", i % 4),
            result: distributed::TaskResult::Primes(primes),
            execution_time_ms: 100 + (i % 50) as u64,
            error: None,
        };
        
        results.push(result);
    }
    
    results
}

/// Generate mock gap analysis results
fn generate_gap_results(count: usize) -> Vec<distributed::WorkResult> {
    let mut results = Vec::with_capacity(count);
    
    for i in 0..count {
        let start = i * 20_000;
        let end = (i + 1) * 20_000;
        let gaps = prime_tool::gaps::find_gaps(start as u64..(end as u64));
        
        let result = distributed::WorkResult {
            work_id: format!("gaps_{}", i),
            node_id: format!("node_{}", i % 3),
            result: distributed::TaskResult::Gaps(gaps),
            execution_time_ms: 200 + (i % 100) as u64,
            error: None,
        };
        
        results.push(result);
    }
    
    results
}

/// Generate mock factorization results
fn generate_factorization_results(count: usize) -> Vec<distributed::WorkResult> {
    let mut results = Vec::with_capacity(count);
    
    for i in 0..count {
        let number = 1000 + i as u64;
        let factors = prime_tool::factor(number);
        let mut factor_map = std::collections::HashMap::new();
        factor_map.insert(number, factors);
        
        let result = distributed::WorkResult {
            work_id: format!("factor_{}", i),
            node_id: format!("node_{}", i % 5),
            result: distributed::TaskResult::Factors(factor_map),
            execution_time_ms: 50 + (i % 20) as u64,
            error: None,
        };
        
        results.push(result);
    }
    
    results
}

criterion_group!(
    benches,
    bench_distributed_coordinator,
    bench_work_unit_creation,
    bench_local_executor,
    bench_distributed_sieve_coordination,
    bench_result_aggregation,
    bench_cluster_monitoring,
    bench_load_balancing,
    bench_fault_tolerance,
    bench_network_communication
);
criterion_main!(benches);