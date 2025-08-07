//! Prime gap analysis and twin prime detection
//!
//! This module provides functionality to analyze gaps between consecutive primes
//! and identify twin prime pairs.

use crate::sieve::sieve_range;
use std::ops::Range;

/// Information about a prime gap
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GapInfo {
    pub start_prime: u64,
    pub end_prime: u64,
    pub gap_size: u32,
    pub is_maximal: bool,
}

/// A twin prime pair
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TwinPrime {
    pub smaller: u64,
    pub larger: u64,
}

/// Find all prime gaps in a given range
///
/// # Examples
///
/// ```
/// use prime_tool::find_gaps;
///
/// let gaps = find_gaps(1..100);
/// assert!(!gaps.is_empty());
/// ```
pub fn find_gaps(range: Range<u64>) -> Vec<GapInfo> {
    let primes = sieve_range(range);
    
    if primes.len() < 2 {
        return vec![];
    }
    
    let mut gaps = Vec::new();
    let mut max_gap_so_far = 0;
    
    for window in primes.windows(2) {
        let start_prime = window[0];
        let end_prime = window[1];
        let gap_size = (end_prime - start_prime) as u32;
        
        let is_maximal = gap_size > max_gap_so_far;
        if is_maximal {
            max_gap_so_far = gap_size;
        }
        
        gaps.push(GapInfo {
            start_prime,
            end_prime,
            gap_size,
            is_maximal,
        });
    }
    
    gaps
}

/// Find all twin prime pairs in a given range
///
/// # Examples
///
/// ```
/// use prime_tool::find_twin_primes;
///
/// let twins = find_twin_primes(1..100);
/// assert!(twins.iter().any(|t| t.smaller == 3 && t.larger == 5));
/// ```
pub fn find_twin_primes(range: Range<u64>) -> Vec<TwinPrime> {
    let primes = sieve_range(range);
    
    let mut twin_primes = Vec::new();
    
    for window in primes.windows(2) {
        let smaller = window[0];
        let larger = window[1];
        
        if larger - smaller == 2 {
            twin_primes.push(TwinPrime { smaller, larger });
        }
    }
    
    twin_primes
}

/// Find cousin prime pairs (gap of 4)
pub fn find_cousin_primes(range: Range<u64>) -> Vec<TwinPrime> {
    let primes = sieve_range(range);
    
    let mut cousin_primes = Vec::new();
    
    for window in primes.windows(2) {
        let smaller = window[0];
        let larger = window[1];
        
        if larger - smaller == 4 {
            cousin_primes.push(TwinPrime { smaller, larger });
        }
    }
    
    cousin_primes
}

/// Find sexy prime pairs (gap of 6)
pub fn find_sexy_primes(range: Range<u64>) -> Vec<TwinPrime> {
    let primes = sieve_range(range);
    
    let mut sexy_primes = Vec::new();
    
    for window in primes.windows(2) {
        let smaller = window[0];
        let larger = window[1];
        
        if larger - smaller == 6 {
            sexy_primes.push(TwinPrime { smaller, larger });
        }
    }
    
    sexy_primes
}

/// Analyze gap distribution and statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GapStatistics {
    pub total_gaps: usize,
    pub min_gap: u32,
    pub max_gap: u32,
    pub average_gap: f64,
    pub median_gap: f64,
    pub gap_histogram: std::collections::HashMap<u32, usize>,
    pub twin_prime_count: usize,
    pub cousin_prime_count: usize,
    pub sexy_prime_count: usize,
}

/// Compute comprehensive gap statistics
pub fn analyze_gaps(range: Range<u64>) -> GapStatistics {
    let gaps = find_gaps(range.clone());
    let twin_primes = find_twin_primes(range.clone());
    let cousin_primes = find_cousin_primes(range.clone());
    let sexy_primes = find_sexy_primes(range);
    
    if gaps.is_empty() {
        return GapStatistics {
            total_gaps: 0,
            min_gap: 0,
            max_gap: 0,
            average_gap: 0.0,
            median_gap: 0.0,
            gap_histogram: std::collections::HashMap::new(),
            twin_prime_count: 0,
            cousin_prime_count: 0,
            sexy_prime_count: 0,
        };
    }
    
    let gap_sizes: Vec<u32> = gaps.iter().map(|g| g.gap_size).collect();
    let min_gap = *gap_sizes.iter().min().unwrap();
    let max_gap = *gap_sizes.iter().max().unwrap();
    let average_gap = gap_sizes.iter().sum::<u32>() as f64 / gap_sizes.len() as f64;
    
    // Calculate median
    let mut sorted_gaps = gap_sizes.clone();
    sorted_gaps.sort_unstable();
    let median_gap = if sorted_gaps.len() % 2 == 0 {
        let mid = sorted_gaps.len() / 2;
        (sorted_gaps[mid - 1] + sorted_gaps[mid]) as f64 / 2.0
    } else {
        sorted_gaps[sorted_gaps.len() / 2] as f64
    };
    
    // Build histogram
    let mut gap_histogram = std::collections::HashMap::new();
    for &gap in &gap_sizes {
        *gap_histogram.entry(gap).or_insert(0) += 1;
    }
    
    GapStatistics {
        total_gaps: gaps.len(),
        min_gap,
        max_gap,
        average_gap,
        median_gap,
        gap_histogram,
        twin_prime_count: twin_primes.len(),
        cousin_prime_count: cousin_primes.len(),
        sexy_prime_count: sexy_primes.len(),
    }
}

/// Find the first occurrence of each gap size
pub fn find_first_gaps(range: Range<u64>) -> std::collections::HashMap<u32, GapInfo> {
    let gaps = find_gaps(range);
    let mut first_gaps = std::collections::HashMap::new();
    
    for gap in gaps {
        first_gaps.entry(gap.gap_size).or_insert(gap);
    }
    
    first_gaps
}

/// Find maximal gaps (gaps larger than all previous gaps)
pub fn find_maximal_gaps(range: Range<u64>) -> Vec<GapInfo> {
    let gaps = find_gaps(range);
    gaps.into_iter().filter(|g| g.is_maximal).collect()
}

/// Bertrand's postulate verification
/// For any integer n > 1, there is always at least one prime p such that n < p < 2n
pub fn verify_bertrand_postulate(range: Range<u64>) -> Vec<u64> {
    let mut violations = Vec::new();
    
    for n in range {
        if n <= 1 {
            continue;
        }
        
        let primes_in_range = sieve_range((n + 1)..(2 * n));
        if primes_in_range.is_empty() {
            violations.push(n);
        }
    }
    
    violations
}

/// Prime gap bounds based on known results
pub fn expected_max_gap(n: u64) -> f64 {
    // Rough approximation based on Cramér's conjecture
    let log_n = (n as f64).ln();
    log_n * log_n
}

/// Check if a gap is unusually large
pub fn is_large_gap(gap: &GapInfo) -> bool {
    let expected = expected_max_gap(gap.start_prime);
    gap.gap_size as f64 > expected * 1.5
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_find_gaps() {
        let gaps = find_gaps(2..20);
        assert!(!gaps.is_empty());
        
        // Check that we have the gap between 2 and 3
        assert!(gaps.iter().any(|g| g.start_prime == 2 && g.end_prime == 3));
    }
    
    #[test]
    fn test_find_twin_primes() {
        let twins = find_twin_primes(1..100);
        
        // Check for known twin primes
        assert!(twins.iter().any(|t| t.smaller == 3 && t.larger == 5));
        assert!(twins.iter().any(|t| t.smaller == 5 && t.larger == 7));
        assert!(twins.iter().any(|t| t.smaller == 11 && t.larger == 13));
    }
    
    #[test]
    fn test_gap_statistics() {
        let stats = analyze_gaps(2..100);
        assert!(stats.total_gaps > 0);
        assert!(stats.min_gap >= 1);
        assert!(stats.max_gap >= stats.min_gap);
        assert!(stats.average_gap > 0.0);
        assert!(stats.twin_prime_count > 0);
    }
    
    #[test]
    fn test_cousin_primes() {
        let cousins = find_cousin_primes(1..100);
        // (3, 7), (7, 11), (13, 17), etc.
        assert!(cousins.iter().any(|c| c.smaller == 3 && c.larger == 7));
    }
    
    #[test]
    fn test_sexy_primes() {
        let sexy = find_sexy_primes(1..100);
        // (5, 11), (7, 13), (11, 17), etc.
        assert!(sexy.iter().any(|s| s.smaller == 5 && s.larger == 11));
    }
    
    #[test]
    fn test_bertrand_postulate() {
        // Bertrand's postulate should hold for reasonable ranges
        let violations = verify_bertrand_postulate(2..1000);
        assert!(violations.is_empty());
    }
    
    #[test]
    fn test_maximal_gaps() {
        let maximal = find_maximal_gaps(2..100);
        assert!(!maximal.is_empty());
        
        // Check that gaps are in increasing order
        for window in maximal.windows(2) {
            assert!(window[1].gap_size > window[0].gap_size);
        }
    }
}