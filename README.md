# 🚀 Prime Research Tool

[![CI](https://github.com/example/prime-research-tool/workflows/CI/badge.svg)](https://github.com/example/prime-research-tool/actions)
[![codecov](https://codecov.io/gh/example/prime-research-tool/branch/main/graph/badge.svg)](https://codecov.io/gh/example/prime-research-tool)
[![Crates.io](https://img.shields.io/crates/v/prime-tool.svg)](https://crates.io/crates/prime-tool)
[![Documentation](https://docs.rs/prime-tool/badge.svg)](https://docs.rs/prime-tool)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

A high-performance Rust command-line tool and library for advanced prime number research, featuring deterministic primality testing, fast factorization, distributed sieving, and comprehensive analysis capabilities.

## ✨ Features

### 🔍 **Primality Testing**
- **Deterministic Miller-Rabin** for numbers up to 2⁶⁴
- **Baillie-PSW** (strong Lucas) available as an independent check
- **Sub-microsecond performance** for candidates ≤ 10⁸
- **Adaptive algorithm selection** based on input size

### 🔧 **Fast Factorization**
- **Pollard's Rho algorithm** for general factorization
- **Brent's method** for improved performance
- **Elliptic Curve Method (ECM)** for 64-bit composites
- **Trial division** optimization for small factors

### 🔢 **Distributed Sieving**
- **Parallel processing** with Rayon
- **Networked distribution** with Tokio
- **Segmented sieving** for memory efficiency
- **Work balancing** across multiple nodes

### 📊 **Analysis & Visualization**
- **Prime gap analysis** (twin, cousin, sexy primes)
- **Statistical analysis** with comprehensive metrics
- **SVG plotting** of π(x) and gap histograms
- **Multiple output formats** (JSON, CSV, tables)

### 🌐 **Web Service**
- **REST API** built with Axum
- **Real-time processing** endpoints
- **Comprehensive documentation** interface
- **Performance monitoring** capabilities

### 📈 **Benchmarking**
- **Criterion-based** performance testing
- **Comprehensive metrics** collection
- **Performance regression** detection
- **Cross-platform** compatibility

## 🚀 Quick Start

### Installation

Requires a Rust toolchain ([rustup.rs](https://rustup.rs)).

```bash
git clone https://github.com/Yinyue93/prime-research-tool.git
cd prime-research-tool

# Build and install the `prime-tool` command into ~/.cargo/bin (on your PATH)
cargo install --path prime-tool-cli
```

Re-run `cargo install --path prime-tool-cli` after making changes to update the installed binary.
To run without installing, use `cargo run --release -p prime-tool-cli -- <args>`.

### Basic Usage

```bash
# Check if a number is prime
prime-tool is-prime -n 999983

# Factor a number
prime-tool factor -n 1234567

# Find all primes in a range
prime-tool sieve -r 1..1000000

# Analyze prime gaps
prime-tool gaps -r 1..100000 --twins

# Generate visualizations (written to ./plots by default)
prime-tool plot -r 1..10000 -p prime-count

# See all commands and options
prime-tool --help
prime-tool <command> --help
```

### Global Options

Global options go **before** the subcommand:

| Option | Description |
|---|---|
| `-f, --format <table\|json\|csv\|plain>` | Output format (default: `table`) |
| `-o, --output <FILE>` | Write results to a file instead of stdout |
| `-j, --threads <N>` | Number of worker threads (default: auto-detect) |
| `-v, --verbose` | Enable debug logging |

```bash
prime-tool -f json is-prime -n 999983     # ✅
prime-tool is-prime -n 999983 -f json     # ❌ error: unexpected argument
```

Ranges are written `start..end` (end exclusive) or `start..=end` (end inclusive).

## 📖 Detailed Usage

### Primality Testing

```bash
prime-tool is-prime -n 999983
prime-tool -f plain is-prime -n 999983
# 999983 is PRIME

# JSON output
prime-tool -f json is-prime -n 999983
# {"number": 999983, "is_prime": true, "certainty": "proven", "execution_time_ms": 0}

# Numbers of any size are accepted
prime-tool is-prime -n 18446744073709551557   # largest 64-bit prime
prime-tool -f plain is-prime -n 170141183460469231731687303715884105727   # 2^127 - 1
# 170141183460469231731687303715884105727 is PROBABLE PRIME
```

Results are **proven** for numbers below ~3.3 × 10²⁴ (deterministic Miller-Rabin).
Above that, primes are reported as **probable primes** (Baillie-PSW: no counterexample
is known, but it is not a proof); "composite" is always proven. The `certainty`
field in JSON/CSV output is `"proven"` or `"probable"`.

### Factorization

```bash
prime-tool -f plain factor -n 1234567
# 1234567 = 127 × 9721

prime-tool factor -n 123456789012345
# 3 × 5 × 283 × 3851 × 7552031

# JSON / CSV output for programmatic use
prime-tool -f json factor -n 1234567
# {"number": 1234567, "factors": [127, 9721], "unfactored": [], "complete": true,
#  "certainty": "proven", "execution_time_ms": 0}
prime-tool -f csv factor -n 1234567

# Numbers larger than 64 bits
prime-tool -f plain factor -n 132543035165161565106540654065014651303
# 132543035165161565106540654065014651303 = 31 × 26641 × 160488787189720386242573784604393

# Limit time spent on hard composites (default 60 s, 0 = no limit)
prime-tool factor -n 340282366920938463463374607431768211457 --timeout 10
```

Factoring numbers above 64 bits uses Pollard-Brent rho, whose running time grows with
the square root of the *second-largest* prime factor. Numbers with factors up to ~15
digits factor quickly; products of two large primes (e.g. 20+ digits each) may not
finish. When the timeout is reached, any unsplit composite parts are listed in
`unfactored` with `complete: false`. JSON numbers are always exact integers, even
beyond 64 bits.

### Sieving Operations

```bash
# Basic range sieving
prime-tool sieve -r 1..1000

# Parallel sieving with 8 threads and a custom chunk size
prime-tool -j 8 sieve -r 1..1000000 --parallel --chunk-size 50000

# Save results to a file
prime-tool -f json -o primes.json sieve -r 1..1000000

# Distributed sieving (requires running worker nodes; repeat --nodes per node)
prime-tool sieve -r 1..10000000 --nodes node1:8080 --nodes node2:8080
```

### Gap Analysis

```bash
# Gap statistics (count, min/max/average/median, twin/cousin/sexy counts)
prime-tool gaps -r 1..100000
# Total gaps: 9591, max gap: 72, average gap: 10.43
# Twin primes: 1224, cousin primes: 1216, sexy primes: 2447

# Include the full lists of twin (p, p+2), cousin (p, p+4) and sexy (p, p+6)
# pairs — use JSON to see the pairs themselves
prime-tool -f json gaps -r 1..100000 --twins --cousins --sexy

# Also write a gap histogram to gap_histogram.svg in the current directory
prime-tool gaps -r 1..100000 --plot
```

### Visualization and Plotting

Plots are SVG files written to `--output-dir` (default: `plots`).

```bash
# Prime counting function π(x)   -> plots/prime_count.svg
prime-tool plot -r 1..10000 -p prime-count

# Gap histogram                   -> plots/gaps.svg
prime-tool plot -r 1..100000 -p gaps

# Prime distribution density      -> plots/distribution.svg
prime-tool plot -r 1..100000 -p distribution

# All plots into a custom directory
prime-tool plot -r 1..100000 -p all --output-dir ./my-plots
```

### Benchmarking

Quick built-in benchmarks (for rigorous measurements, use `cargo bench` — see below).

```bash
# Run all benchmarks (defaults: --max-n 100000000 --iterations 1000)
prime-tool benchmark

# Specific benchmark types: primality, factorization, sieve
prime-tool benchmark -b primality --max-n 1000000 --iterations 10000
prime-tool benchmark -b factorization --max-n 1000000 --iterations 1000
prime-tool benchmark -b sieve --max-n 10000000 --iterations 5
```

### Web Service

The REST API is a separate binary in the `prime-tool-web` crate and listens on
`127.0.0.1:3000`. (The `prime-tool web` subcommand is currently a placeholder
and does not start a server.)

```bash
cargo run --release -p prime-tool-web

# or install it alongside the CLI
cargo install --path prime-tool-web
prime-tool-web
```

#### Web API Endpoints

```bash
# Health check
curl http://localhost:3000/health

# Check primality
curl http://localhost:3000/prime/999983
curl -X POST http://localhost:3000/prime -H "Content-Type: application/json" -d '{"numbers": [999983, 1000003]}'

# Factorization
curl http://localhost:3000/factor/1234567
curl -X POST http://localhost:3000/factor -H "Content-Type: application/json" -d '{"numbers": [1234567, 9876543]}'

# Sieving
curl "http://localhost:3000/sieve?start=1&end=1000"
curl "http://localhost:3000/sieve/parallel?start=1&end=1000000&chunk_size=10000"

# Gap analysis
curl "http://localhost:3000/gaps?start=1&end=10000"
curl "http://localhost:3000/gaps/twins?start=1&end=100000"
curl "http://localhost:3000/gaps/cousins?start=1&end=100000"
curl "http://localhost:3000/gaps/sexy?start=1&end=100000"

# Statistics
curl "http://localhost:3000/stats/range?start=1&end=100000"

# Benchmarking
curl -X POST http://localhost:3000/benchmark -H "Content-Type: application/json" -d '{"max_n": 100000, "iterations": 1000, "benchmark_type": "primality"}'
```

## 📚 Library API

### Basic Usage

```rust
use prime_tool::*;

fn main() {
    // Primality testing
    assert!(is_prime(999983));
    assert!(!is_prime(999984));
    
    // Factorization
    let factors = factor(1234567);
    println!("Factors of 1234567: {:?}", factors); // [127, 9721]
    
    // Sieving
    let primes = sieve_range(1..1000);
    println!("Found {} primes", primes.len()); // 168
    
    // Gap analysis
    let gaps = gaps::find_gaps(1..10000);
    let twins = gaps::find_twin_primes(1..10000);
    println!("Found {} gaps, {} twin pairs", gaps.len(), twins.len());
}
```

### Advanced Usage

```rust
use prime_tool::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure for high-performance operation
    let config = Config {
        max_threads: 16,
        chunk_size: 100_000,
        enable_distributed: true,
        ..Default::default()
    };
    
    // Parallel sieving
    let primes = sieve_parallel(1..10_000_000, config.chunk_size);
    println!("Found {} primes using parallel sieve", primes.len());
    
    // Distributed computing setup
    let mut coordinator = distributed::DistributedCoordinator::new();
    coordinator.register_node(distributed::NodeInfo {
        id: "node1".to_string(),
        address: "192.168.1.100:8080".to_string(),
        capabilities: vec!["sieve".to_string(), "primality".to_string()],
        load: 0.0,
    });
    
    // Distributed sieving
    let distributed_primes = coordinator.distributed_sieve(1..100_000_000, 8).await?;
    println!("Distributed sieve found {} primes", distributed_primes.len());
    
    // Comprehensive gap analysis
    let gap_stats = gaps::analyze_gaps(1..1_000_000);
    println!("Gap statistics: max={}, avg={:.2}, total={}", 
             gap_stats.max_gap, gap_stats.average_gap, gap_stats.total_gaps);
    
    // Performance monitoring
    let results = PrimeResults {
        primes: primes.clone(),
        statistics: Statistics {
            total_candidates: 10_000_000,
            primes_found: primes.len(),
            largest_prime: *primes.last().unwrap_or(&0),
            processing_time: std::time::Duration::from_millis(1500),
            memory_used: 50_000_000, // bytes
        },
        gaps: Some(gaps::find_gaps(1..10_000_000)),
        twin_primes: Some(gaps::find_twin_primes(1..10_000_000)),
    };
    
    println!("Processing completed: {} primes in {:.2}s", 
             results.statistics.primes_found,
             results.statistics.processing_time.as_secs_f64());
    
    Ok(())
}
```

### Custom Algorithm Implementation

```rust
use prime_tool::algorithms::*;

// Custom primality test with specific witnesses
fn custom_miller_rabin(n: u64) -> bool {
    let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    miller_rabin(n, &witnesses)
}

// Custom factorization with timeout
fn factor_with_timeout(n: u64, timeout_ms: u64) -> Vec<u64> {
    let start = std::time::Instant::now();
    let mut factors = Vec::new();
    let mut remaining = n;
    
    // Trial division for small factors
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        while remaining % p == 0 {
            factors.push(p);
            remaining /= p;
        }
        if start.elapsed().as_millis() > timeout_ms as u128 {
            break;
        }
    }
    
    // Use advanced methods for remaining composite
    if remaining > 1 && start.elapsed().as_millis() <= timeout_ms as u128 {
        if let Some(factor) = factorization::pollard_rho(remaining) {
            factors.push(factor);
            factors.push(remaining / factor);
        } else {
            factors.push(remaining);
        }
    }
    
    factors.sort();
    factors
}
```

## 🏗️ Architecture

### Workspace Structure

```
prime-research-tool/
├── prime-tool/           # Core library
│   ├── src/
│   │   ├── algorithms.rs # Primality testing algorithms
│   │   ├── factorization.rs # Factorization methods
│   │   ├── sieve.rs      # Sieving algorithms
│   │   ├── gaps.rs       # Gap analysis
│   │   ├── distributed.rs # Distributed computing
│   │   └── error.rs      # Error handling
│   └── benches/          # Criterion benchmarks
├── prime-tool-cli/       # Command-line interface
│   └── src/
│       ├── main.rs       # CLI argument parsing
│       ├── output.rs     # Output formatting
│       ├── plotting.rs   # SVG plot generation
│       └── benchmarks.rs # Performance testing
├── prime-tool-web/       # Web service (Axum)
│   └── src/
│       └── main.rs       # REST API endpoints
├── prime-tool-bench/     # Comprehensive benchmarking
│   └── src/
│       └── main.rs       # Benchmark suite
└── .github/workflows/    # CI/CD configuration
```

### Performance Characteristics

| Operation | Range | Performance | Memory |
|-----------|-------|-------------|--------|
| Primality Test | ≤ 10⁸ | < 1μs | O(1) |
| Factorization | ≤ 10¹² | < 100ms | O(log n) |
| Sieving | ≤ 10⁹ | 1M primes/s | O(n/log n) |
| Gap Analysis | ≤ 10⁷ | 100K gaps/s | O(π(n)) |
| Distributed | ≤ 10¹⁰ | Linear scaling | O(n/nodes) |

## 🧪 Testing

```bash
# Run all tests
cargo test --workspace

# Run with coverage
cargo install cargo-llvm-cov
cargo llvm-cov --workspace --html

# Run benchmarks
cd prime-tool
cargo bench

# Run comprehensive benchmark suite
cd prime-tool-bench
cargo run --release

# Integration tests
cargo test --test integration_tests

# Doctests
cargo test --doc
```

## 📊 Benchmarks

Performance results on Intel i7-12700K @ 3.6GHz:

### Primality Testing
- **Small primes (< 10⁶)**: 0.1-0.5μs average
- **Medium primes (10⁶-10⁹)**: 0.5-2.0μs average  
- **Large primes (> 10⁹)**: 2.0-10μs average
- **Throughput**: > 1M tests/second

### Factorization
- **Small semiprimes**: < 1ms average
- **Medium semiprimes**: 1-50ms average
- **Large semiprimes**: 50-500ms average
- **Success rate**: > 99% within timeout

### Sieving
- **Sequential**: 500K-1M primes/second
- **Parallel (8 cores)**: 3-4M primes/second
- **Memory efficiency**: ~8 bytes per prime
- **Scalability**: Linear with core count

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Yinyue93/prime-research-tool.git
cd prime-research-tool

# Install development dependencies
cargo install cargo-watch cargo-audit cargo-llvm-cov

# Run tests in watch mode
cargo watch -x test

# Format code
cargo fmt

# Run clippy
cargo clippy --workspace --all-targets --all-features

# Security audit
cargo audit
```

### Code Style

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting
- Ensure all `clippy` warnings are addressed
- Write comprehensive tests and documentation
- Maintain performance benchmarks

## 📄 License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙏 Acknowledgments

- **Miller-Rabin Algorithm**: Gary L. Miller, Michael O. Rabin
- **Baillie-PSW Test**: Robert Baillie, Carl Pomerance, John Selfridge, Samuel Wagstaff
- **Pollard's Rho**: John Pollard
- **Elliptic Curve Method**: Hendrik Lenstra
- **Rust Community**: For excellent crates and tooling

## 📞 Support

- 📖 [Documentation](https://docs.rs/prime-tool)
- 🐛 [Issue Tracker](https://github.com/example/prime-research-tool/issues)
- 💬 [Discussions](https://github.com/example/prime-research-tool/discussions)
- 📧 Email: prime-tool@example.com

---

**Built with ❤️ in Rust** | **Performance Meets Elegance**