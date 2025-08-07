//! Criterion benchmarks for distributed computing functionality

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prime_tool::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Benchmark distributed coordinator overhead
fn bench_distributed_coordinator(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_coordinator");
    
    let rt = Runtime::new().unwrap();
    
    // Node registration overhead
    group.bench_function("node_registration", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut coordinator = black_box(distributed::DistributedCoordinator::new());
                
                // Register multiple nodes
                for i in 0..10 {
                    let node_info = distributed::NodeInfo {
                        id: format!("node_{}", i),
                        address: format!("127.0.0.1:{}", 8000 + i),
                        capabilities: vec!["sieve".to_string(), "factor".to_string()],
                        max_concurrent_tasks: 4,
                    };
                    
                    black_box(coordinator.register_node(black_box(node_info)).await);
                }
                
                black_box(coordinator.active_nodes());
            });
        });
    });
    
    // Work unit creation and distribution
    group.bench_function("work_distribution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut coordinator = black_box(distributed::DistributedCoordinator::new());
                
                // Register nodes
                for i in 0..4 {
                    let node_info = distributed::NodeInfo {
                        id: format!("node_{}", i),
                        address: format!("127.0.0.1:{}", 8000 + i),
                        capabilities: vec!["sieve".to_string()],
                        max_concurrent_tasks: 2,
                    };
                    coordinator.register_node(node_info).await;
                }
                
                // Submit work
                let work_unit = distributed::WorkUnit {
                    id: "test_work".to_string(),
                    task: distributed::TaskType::PrimeSieve {
                        range: 2..1_000_000,
                    },
                    priority: 1,
                    estimated_duration: std::time::Duration::from_secs(10),
                };
                
                black_box(coordinator.submit_work(black_box(work_unit)).await);
                
                // Assign work to nodes
                for _ in 0..4 {
                    if let Some(assignment) = coordinator.assign_work().await {
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
        
        // Gap analysis work units
        group.bench_with_input(
            BenchmarkId::new("gap_work_units", config_name),
            &range,
            |b, range| {
                b.iter(|| {
                    let work_units = black_box(distributed::create_gap_work_units(
                        black_box(range.clone()),
                        black_box(4), // 4 workers
                    ));
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
                let executor = black_box(distributed::LocalExecutor::new());
                
                let work_unit = distributed::WorkUnit {
                    id: "sieve_test".to_string(),
                    task: distributed::TaskType::PrimeSieve {
                        range: 2..100_000,
                    },
                    priority: 1,
                    estimated_duration: std::time::Duration::from_secs(1),
                };
                
                let result = black_box(executor.execute_work(black_box(work_unit)).await);
                black_box(result);
            });
        });
    });
    
    // Factorization execution
    group.bench_function("local_factorization_execution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let executor = black_box(distributed::LocalExecutor::new());
                
                let work_unit = distributed::WorkUnit {
                    id: "factor_test".to_string(),
                    task: distributed::TaskType::Factorization {
                        number: 982451653 * 982451657,
                    },
                    priority: 1,
                    estimated_duration: std::time::Duration::from_millis(100),
                };
                
                let result = black_box(executor.execute_work(black_box(work_unit)).await);
                black_box(result);
            });
        });
    });
    
    // Primality test execution
    group.bench_function("local_primality_execution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let executor = black_box(distributed::LocalExecutor::new());
                
                let work_unit = distributed::WorkUnit {
                    id: "primality_test".to_string(),
                    task: distributed::TaskType::PrimalityTest {
                        number: 982451653,
                    },
                    priority: 1,
                    estimated_duration: std::time::Duration::from_micros(100),
                };
                
                let result = black_box(executor.execute_work(black_box(work_unit)).await);
                black_box(result);
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
                        let result = black_box(distributed::distributed_sieve(
                            black_box(test_range.clone()),
                            black_box(workers),
                        ).await);
                        black_box(result.len());
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
                let aggregated = black_box(distributed::aggregate_sieve_results(
                    black_box(&sieve_results)
                ).await);
                black_box(aggregated.len());
            });
        });
    });
    
    // Gap analysis result aggregation
    group.throughput(Throughput::Elements(gap_results.len() as u64));
    group.bench_function("gap_aggregation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let aggregated = black_box(distributed::aggregate_gap_results(
                    black_box(&gap_results)
                ).await);
                black_box(aggregated.total_gaps);
            });
        });
    });
    
    // Factorization result aggregation
    group.throughput(Throughput::Elements(factorization_results.len() as u64));
    group.bench_function("factorization_aggregation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let aggregated = black_box(distributed::aggregate_factorization_results(
                    black_box(&factorization_results)
                ).await);
                black_box(aggregated.len());
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
                let mut coordinator = black_box(distributed::DistributedCoordinator::new());
                
                // Register nodes with different loads
                for i in 0..10 {
                    let node_info = distributed::NodeInfo {
                        id: format!("node_{}", i),
                        address: format!("127.0.0.1:{}", 8000 + i),
                        capabilities: vec!["sieve".to_string(), "factor".to_string()],
                        max_concurrent_tasks: 4,
                    };
                    coordinator.register_node(node_info).await;
                }
                
                // Submit various work units
                for j in 0..20 {
                    let work_unit = distributed::WorkUnit {
                        id: format!("work_{}", j),
                        task: distributed::TaskType::PrimeSieve {
                            range: (j * 10_000)..((j + 1) * 10_000),
                        },
                        priority: 1,
                        estimated_duration: std::time::Duration::from_secs(1),
                    };
                    coordinator.submit_work(work_unit).await;
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
            |b, strategy_name| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut coordinator = black_box(distributed::DistributedCoordinator::new());
                        coordinator.set_load_balancing_strategy(strategy_name);
                        
                        // Register nodes with different capabilities
                        for i in 0..8 {
                            let capabilities = match i % 3 {
                                0 => vec!["sieve".to_string()],
                                1 => vec!["factor".to_string()],
                                _ => vec!["sieve".to_string(), "factor".to_string(), "gaps".to_string()],
                            };
                            
                            let node_info = distributed::NodeInfo {
                                id: format!("node_{}", i),
                                address: format!("127.0.0.1:{}", 8000 + i),
                                capabilities,
                                max_concurrent_tasks: 2 + (i % 3),
                            };
                            coordinator.register_node(node_info).await;
                        }
                        
                        // Submit mixed workload
                        let work_types = vec![
                            distributed::TaskType::PrimeSieve { range: 2..100_000 },
                            distributed::TaskType::Factorization { number: 982451653 },
                            distributed::TaskType::GapAnalysis { range: 2..50_000 },
                            distributed::TaskType::PrimalityTest { number: 999983 },
                        ];
                        
                        for (i, task) in work_types.into_iter().enumerate() {
                            let work_unit = distributed::WorkUnit {
                                id: format!("mixed_work_{}", i),
                                task,
                                priority: (i % 3) + 1,
                                estimated_duration: std::time::Duration::from_secs(1),
                            };
                            coordinator.submit_work(work_unit).await;
                        }
                        
                        // Assign work using the strategy
                        let mut assignments = 0;
                        while let Some(assignment) = coordinator.assign_work().await {
                            black_box(assignment);
                            assignments += 1;
                            if assignments >= 10 {
                                break;
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
                let mut coordinator = black_box(distributed::DistributedCoordinator::new());
                
                // Register nodes
                for i in 0..5 {
                    let node_info = distributed::NodeInfo {
                        id: format!("node_{}", i),
                        address: format!("127.0.0.1:{}", 8000 + i),
                        capabilities: vec!["sieve".to_string()],
                        max_concurrent_tasks: 2,
                    };
                    coordinator.register_node(node_info).await;
                }
                
                // Submit work
                for i in 0..10 {
                    let work_unit = distributed::WorkUnit {
                        id: format!("work_{}", i),
                        task: distributed::TaskType::PrimeSieve {
                            range: (i * 10_000)..((i + 1) * 10_000),
                        },
                        priority: 1,
                        estimated_duration: std::time::Duration::from_secs(5),
                    };
                    coordinator.submit_work(work_unit).await;
                }
                
                // Assign work
                let mut assignments = Vec::new();
                for _ in 0..5 {
                    if let Some(assignment) = coordinator.assign_work().await {
                        assignments.push(assignment);
                    }
                }
                
                // Simulate node failure
                coordinator.handle_node_failure("node_2").await;
                
                // Reassign failed work
                let reassigned = black_box(coordinator.reassign_failed_work().await);
                black_box(reassigned);
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
                        let primes: Vec<u64> = (2..size).filter(|&n| is_prime(n)).collect();
                        
                        // Serialize
                        let serialized = black_box(distributed::serialize_result(&primes).await);
                        
                        // Deserialize
                        let deserialized: Vec<u64> = black_box(distributed::deserialize_result(&serialized).await);
                        
                        black_box(deserialized.len());
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
        let primes = sieve_range(start as u64..(end as u64));
        
        let result = distributed::WorkResult {
            work_id: format!("sieve_{}", i),
            node_id: format!("node_{}", i % 4),
            result: distributed::TaskResult::SieveResult { primes },
            execution_time: std::time::Duration::from_millis(100 + (i % 50) as u64),
            success: true,
            error_message: None,
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
        let gaps = gaps::find_gaps(start as u64..(end as u64));
        
        let result = distributed::WorkResult {
            work_id: format!("gaps_{}", i),
            node_id: format!("node_{}", i % 3),
            result: distributed::TaskResult::GapResult { gaps },
            execution_time: std::time::Duration::from_millis(200 + (i % 100) as u64),
            success: true,
            error_message: None,
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
        let factors = factor(number);
        
        let result = distributed::WorkResult {
            work_id: format!("factor_{}", i),
            node_id: format!("node_{}", i % 5),
            result: distributed::TaskResult::FactorizationResult { factors },
            execution_time: std::time::Duration::from_micros(50 + (i % 20) as u64),
            success: true,
            error_message: None,
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