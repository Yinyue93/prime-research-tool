//! Prime sieving algorithms with parallel and distributed support
//!
//! This module implements efficient prime sieving using the Sieve of Eratosthenes
//! with optimizations for parallel processing across multiple cores and network nodes.

use crate::algorithms::is_prime;
use crate::error::Result;
use rayon::prelude::*;
use std::ops::Range;
use tokio::sync::mpsc;

/// Find all primes in a given range
///
/// # Examples
///
/// ```
/// use prime_tool::sieve_range;
///
/// let primes = sieve_range(1..100);
/// assert!(primes.contains(&97));
/// assert!(!primes.contains(&98));
/// ```
pub fn sieve_range(range: Range<u64>) -> Vec<u64> {
    if range.start >= range.end {
        return vec![];
    }
    
    let start = range.start.max(2);
    let end = range.end;
    
    if end - start < 1000 {
        // Use simple trial division for small ranges
        return (start..end).filter(|&n| is_prime(n)).collect();
    }
    
    // Use segmented sieve for larger ranges
    segmented_sieve_impl(start, end)
}

/// Parallel prime sieving using Rayon
pub fn sieve_parallel(range: Range<u64>, chunk_size: usize) -> Vec<u64> {
    if range.start >= range.end {
        return vec![];
    }
    
    let start = range.start.max(2);
    let end = range.end;
    
    if end - start < chunk_size as u64 {
        return sieve_range(start..end);
    }
    
    // Split range into chunks and process in parallel
    let chunks: Vec<Range<u64>> = (start..end)
        .step_by(chunk_size)
        .map(|chunk_start| {
            let chunk_end = (chunk_start + chunk_size as u64).min(end);
            chunk_start..chunk_end
        })
        .collect();
    
    let mut all_primes: Vec<u64> = chunks
        .par_iter()
        .flat_map(|chunk| sieve_range(chunk.clone()))
        .collect();
    
    all_primes.sort_unstable();
    all_primes
}

/// Parallel sieve with default chunk size
pub fn parallel_sieve(range: Range<u64>) -> Vec<u64> {
    sieve_parallel(range, 10000)
}

/// Segmented sieve with Range input
pub fn segmented_sieve(range: Range<u64>) -> Vec<u64> {
    segmented_sieve_impl(range.start, range.end)
}

/// Segmented sieve with custom segment size
pub fn segmented_sieve_with_size(range: Range<u64>, segment_size: u64) -> Vec<u64> {
    if range.start >= range.end {
        return vec![];
    }
    
    let start = range.start.max(2);
    let end = range.end;
    
    // First, find all primes up to sqrt(end)
    let limit = ((end as f64).sqrt() as u64) + 1;
    let base_primes = simple_sieve(limit);
    
    let mut primes = Vec::new();
    
    // Add small primes that are in range
    for &p in &base_primes {
        if p >= start && p < end {
            primes.push(p);
        }
    }
    
    // Process in custom-sized segments
    let mut current = start.max(limit);
    
    while current < end {
        let segment_end = (current + segment_size).min(end);
        let segment_primes = sieve_segment(current, segment_end, &base_primes);
        primes.extend(segment_primes);
        current = segment_end;
    }
    
    primes.sort_unstable();
    primes
}

/// Count primes in range without storing them
pub fn count_primes_in_range(range: Range<u64>) -> usize {
    if range.start >= range.end {
        return 0;
    }
    
    let start = range.start.max(2);
    let end = range.end;
    
    if end - start < 1000 {
        return (start..end).filter(|&n| is_prime(n)).count();
    }
    
    // Use segmented sieve but only count
    let limit = ((end as f64).sqrt() as u64) + 1;
    let base_primes = simple_sieve(limit);
    
    let mut count = 0;
    
    // Count small primes that are in range
    for &p in &base_primes {
        if p >= start && p < end {
            count += 1;
        }
    }
    
    // Process in segments and count
    const SEGMENT_SIZE: u64 = 1_000_000;
    let mut current = start.max(limit);
    
    while current < end {
        let segment_end = (current + SEGMENT_SIZE).min(end);
        let segment_primes = sieve_segment(current, segment_end, &base_primes);
        count += segment_primes.len();
        current = segment_end;
    }
    
    count
}

/// Basic sieve implementation
pub fn basic_sieve(range: Range<u64>) -> Vec<u64> {
    if range.start >= range.end {
        return vec![];
    }
    
    let start = range.start.max(2);
    let end = range.end;
    
    if end <= start {
        return vec![];
    }
    
    let size = (end - start) as usize;
    let mut is_prime = vec![true; size];
    
    for i in 2..((end as f64).sqrt() as u64 + 1) {
        if i >= start {
            let idx = (i - start) as usize;
            if idx < is_prime.len() && !is_prime[idx] {
                continue;
            }
        }
        
        let mut multiple = ((start + i - 1) / i) * i;
        if multiple < i * i {
            multiple = i * i;
        }
        
        while multiple < end {
            if multiple >= start {
                is_prime[(multiple - start) as usize] = false;
            }
            multiple += i;
        }
    }
    
    (start..end)
        .enumerate()
        .filter_map(|(i, n)| {
            if is_prime[i] {
                Some(n)
            } else {
                None
            }
        })
        .collect()
}

/// Optimized sieve (skips even numbers)
pub fn optimized_sieve(range: Range<u64>) -> Vec<u64> {
    if range.start >= range.end {
        return vec![];
    }
    
    let start = range.start.max(2);
    let end = range.end;
    
    let mut primes = Vec::new();
    
    // Add 2 if in range
    if start <= 2 && 2 < end {
        primes.push(2);
    }
    
    // Only check odd numbers
    let odd_start = if start % 2 == 0 { start + 1 } else { start };
    
    for n in (odd_start..end).step_by(2) {
        if is_prime(n) {
            primes.push(n);
        }
    }
    
    primes
}

/// Wheel sieve using 2,3 wheel
pub fn wheel_sieve(range: Range<u64>) -> Vec<u64> {
    let wheel = WheelSieve::new(&[2, 3]);
    let candidates = wheel.candidates(range.start, range.end);
    candidates.into_iter().filter(|&n| is_prime(n)).collect()
}

/// Wheel factorization sieve
pub fn wheel_factorization_sieve(range: Range<u64>) -> Vec<u64> {
    let wheel = WheelSieve::new(&[2, 3, 5]);
    let candidates = wheel.candidates(range.start, range.end);
    candidates.into_iter().filter(|&n| is_prime(n)).collect()
}

/// BitVec-based sieve
pub fn bitvec_sieve(range: Range<u64>) -> Vec<u64> {
    // For now, use basic sieve (could be optimized with actual bitvec crate)
    basic_sieve(range)
}

/// Vec<bool>-based sieve
pub fn vec_bool_sieve(range: Range<u64>) -> Vec<u64> {
    basic_sieve(range)
}

/// Compressed sieve (odd numbers only)
pub fn compressed_sieve(range: Range<u64>) -> Vec<u64> {
    optimized_sieve(range)
}

/// SIMD-optimized sieve (x86_64 only)
#[cfg(target_arch = "x86_64")]
pub fn simd_sieve(range: Range<u64>) -> Vec<u64> {
    // For now, fallback to basic sieve
    basic_sieve(range)
}

/// SIMD-optimized sieve fallback for non-x86_64
#[cfg(not(target_arch = "x86_64"))]
pub fn simd_sieve(range: Range<u64>) -> Vec<u64> {
    // Fallback to basic sieve on non-x86_64 architectures
    basic_sieve(range)
}

/// Segmented Sieve of Eratosthenes
fn segmented_sieve_impl(start: u64, end: u64) -> Vec<u64> {
    if start >= end {
        return vec![];
    }
    
    // First, find all primes up to sqrt(end)
    let limit = ((end as f64).sqrt() as u64) + 1;
    let base_primes = simple_sieve(limit);
    
    let mut primes = Vec::new();
    
    // Add small primes that are in range
    for &p in &base_primes {
        if p >= start && p < end {
            primes.push(p);
        }
    }
    
    // Process in segments
    const SEGMENT_SIZE: u64 = 1_000_000;
    let mut current = start.max(limit);
    
    while current < end {
        let segment_end = (current + SEGMENT_SIZE).min(end);
        let segment_primes = sieve_segment(current, segment_end, &base_primes);
        primes.extend(segment_primes);
        current = segment_end;
    }
    
    primes.sort_unstable();
    primes
}

/// Simple Sieve of Eratosthenes for small ranges
fn simple_sieve(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return vec![];
    }
    
    let mut is_prime = vec![true; limit as usize];
    is_prime[0] = false;
    if limit > 1 {
        is_prime[1] = false;
    }
    
    let sqrt_limit = (limit as f64).sqrt() as usize;
    
    for i in 2..=sqrt_limit {
        if is_prime[i] {
            let mut j = i * i;
            while j < limit as usize {
                is_prime[j] = false;
                j += i;
            }
        }
    }
    
    (2..limit)
        .filter(|&i| is_prime[i as usize])
        .collect()
}

/// Sieve a segment using known base primes
fn sieve_segment(start: u64, end: u64, base_primes: &[u64]) -> Vec<u64> {
    if start >= end {
        return vec![];
    }
    
    let size = (end - start) as usize;
    let mut is_prime = vec![true; size];
    
    for &p in base_primes {
        if p * p > end {
            break;
        }
        
        // Find the first multiple of p >= start
        let mut first_multiple = ((start + p - 1) / p) * p;
        if first_multiple == p {
            first_multiple = p * p;
        }
        
        // Mark multiples as composite
        let mut multiple = first_multiple;
        while multiple < end {
            if multiple >= start {
                is_prime[(multiple - start) as usize] = false;
            }
            multiple += p;
        }
    }
    
    (start..end)
        .enumerate()
        .filter_map(|(i, n)| {
            if is_prime[i] && n >= 2 {
                Some(n)
            } else {
                None
            }
        })
        .collect()
}

/// Distributed sieving coordinator
pub struct DistributedSieve {
    nodes: Vec<String>,
    chunk_size: u64,
}

impl DistributedSieve {
    pub fn new(nodes: Vec<String>, chunk_size: u64) -> Self {
        Self { nodes, chunk_size }
    }
    
    /// Distribute sieving work across network nodes
    pub async fn sieve_distributed(&self, range: Range<u64>) -> Result<Vec<u64>> {
        if self.nodes.is_empty() {
            return Ok(sieve_parallel(range, self.chunk_size as usize));
        }
        
        let (tx, mut rx) = mpsc::channel(100);
        let total_range = range.end - range.start;
        let chunk_size = (total_range / self.nodes.len() as u64).max(self.chunk_size);
        
        // Spawn tasks for each node
        let mut handles = Vec::new();
        
        for (i, node) in self.nodes.iter().enumerate() {
            let node_start = range.start + i as u64 * chunk_size;
            let node_end = (node_start + chunk_size).min(range.end);
            
            if node_start >= range.end {
                break;
            }
            
            let node_range = node_start..node_end;
            let node_url = node.clone();
            let tx_clone = tx.clone();
            
            let handle = tokio::spawn(async move {
                match sieve_remote_node(&node_url, node_range).await {
                    Ok(primes) => {
                        let _ = tx_clone.send(Ok(primes)).await;
                    }
                    Err(e) => {
                        let _ = tx_clone.send(Err(e)).await;
                    }
                }
            });
            
            handles.push(handle);
        }
        
        drop(tx); // Close the sender
        
        // Collect results
        let mut all_primes = Vec::new();
        
        while let Some(result) = rx.recv().await {
            match result {
                Ok(primes) => all_primes.extend(primes),
                Err(e) => {
                    log::warn!("Node failed: {}", e);
                    // Continue with other nodes
                }
            }
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }
        
        all_primes.sort_unstable();
        Ok(all_primes)
    }
}

/// Send sieving work to a remote node
async fn sieve_remote_node(node_url: &str, range: Range<u64>) -> Result<Vec<u64>> {
    // This would implement actual network communication
    // For now, we'll simulate with local computation
    log::info!("Sieving range {}..{} on node {}", range.start, range.end, node_url);
    
    // Simulate network delay
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    
    // Fallback to local computation
    Ok(sieve_range(range))
}

/// Optimized wheel sieve for better performance
pub struct WheelSieve {
    wheel: Vec<u64>,
    wheel_size: u64,
}

impl WheelSieve {
    /// Create a new wheel sieve with the given base primes
    pub fn new(base_primes: &[u64]) -> Self {
        let wheel_size = base_primes.iter().product();
        let mut wheel = Vec::new();
        
        for i in 1..wheel_size {
            if base_primes.iter().all(|&p| i % p != 0) {
                wheel.push(i);
            }
        }
        
        Self { wheel, wheel_size }
    }
    
    /// Generate candidates using the wheel
    pub fn candidates(&self, start: u64, end: u64) -> Vec<u64> {
        let mut candidates = Vec::new();
        
        let wheel_start = (start / self.wheel_size) * self.wheel_size;
        let mut current_wheel = wheel_start;
        
        while current_wheel < end {
            for &offset in &self.wheel {
                let candidate = current_wheel + offset;
                if candidate >= start && candidate < end {
                    candidates.push(candidate);
                }
            }
            current_wheel += self.wheel_size;
        }
        
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_sieve() {
        let primes = simple_sieve(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }
    
    #[test]
    fn test_sieve_range() {
        let primes = sieve_range(10..20);
        assert_eq!(primes, vec![11, 13, 17, 19]);
    }
    
    #[test]
    fn test_segmented_sieve() {
        let primes = segmented_sieve(100..200);
        assert!(primes.contains(&101));
        assert!(primes.contains(&199));
        assert!(!primes.contains(&100));
    }
    
    #[test]
    fn test_parallel_sieve() {
        let primes = sieve_parallel(1..100, 10);
        assert!(primes.contains(&97));
        assert_eq!(primes.len(), 25); // 25 primes under 100
    }
    
    #[test]
    fn test_wheel_sieve() {
        let wheel = WheelSieve::new(&[2, 3]);
        let candidates = wheel.candidates(1, 30);
        // Should skip multiples of 2 and 3
        assert!(!candidates.contains(&4));
        assert!(!candidates.contains(&6));
        assert!(!candidates.contains(&9));
    }
}