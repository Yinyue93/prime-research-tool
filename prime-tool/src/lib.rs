//! Prime Research Tool - High-performance prime number research library
//!
//! This library provides advanced algorithms for:
//! - Deterministic primality testing using Miller-Rabin and Baillie-PSW
//! - Fast factorization with Pollard's Rho, Brent's method, and elliptic curves
//! - Distributed prime sieving across multiple cores and network nodes
//! - Prime gap analysis and twin prime detection
//!
//! # Examples
//!
//! ```
//! use prime_tool::{is_prime, factor, sieve_range};
//!
//! // Check if a number is prime
//! assert!(is_prime(999983));
//! assert!(!is_prime(999984));
//!
//! // Factor a composite number
//! let factors = factor(999984);
//! assert!(!factors.is_empty());
//!
//! // Find all primes in a range
//! let primes = sieve_range(1..100);
//! assert!(primes.contains(&97));
//! ```

pub mod algorithms;
pub mod factorization;
pub mod sieve;
pub mod gaps;
pub mod distributed;
pub mod error;

pub use algorithms::{is_prime, miller_rabin, baillie_psw};
pub use factorization::{factor, pollard_rho, brent_factor, ecm_factor};
pub use sieve::{sieve_range, sieve_parallel};
pub use gaps::{find_gaps, find_twin_primes, GapInfo, TwinPrime};
pub use error::{PrimeError, Result};

/// Prime number research results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrimeResults {
    pub primes: Vec<u64>,
    pub gaps: Vec<GapInfo>,
    pub twin_primes: Vec<TwinPrime>,
    pub statistics: Statistics,
}

/// Statistical information about prime analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Statistics {
    pub total_candidates: u64,
    pub primes_found: u64,
    pub largest_gap: u32,
    pub average_gap: f64,
    pub twin_prime_count: u64,
    pub processing_time_ms: u64,
}

/// Configuration for prime research operations
#[derive(Debug, Clone)]
pub struct Config {
    pub parallel: bool,
    pub thread_count: Option<usize>,
    pub chunk_size: usize,
    pub use_distributed: bool,
    pub network_nodes: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            parallel: true,
            thread_count: None,
            chunk_size: 10_000,
            use_distributed: false,
            network_nodes: Vec::new(),
        }
    }
}