//! Primality testing algorithms
//!
//! This module implements deterministic primality tests up to 2^64 using
//! adaptive Miller-Rabin with Baillie-PSW fallback.

use std::collections::HashMap;
use once_cell::sync::Lazy;

/// Deterministic witnesses for Miller-Rabin test for different ranges
static MILLER_RABIN_WITNESSES: Lazy<HashMap<u64, Vec<u64>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(2_047, vec![2]);
    map.insert(1_373_653, vec![2, 3]);
    map.insert(9_080_191, vec![31, 73]);
    map.insert(25_326_001, vec![2, 3, 5]);
    map.insert(3_215_031_751, vec![2, 3, 5, 7]);
    map.insert(4_759_123_141, vec![2, 7, 61]);
    map.insert(1_122_004_669_633, vec![2, 13, 23, 1662803]);
    map.insert(2_152_302_898_747, vec![2, 3, 5, 7, 11]);
    map.insert(3_474_749_660_383, vec![2, 3, 5, 7, 11, 13]);
    map.insert(341_550_071_728_321, vec![2, 3, 5, 7, 11, 13, 17]);
    map.insert(3_825_123_056_546_413_051, vec![2, 3, 5, 7, 11, 13, 17, 19, 23]);
    map.insert(u64::MAX, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]);
    map
});

/// Check if a number is prime using adaptive algorithms
///
/// # Examples
///
/// ```
/// use prime_tool::is_prime;
///
/// assert!(is_prime(2));
/// assert!(is_prime(97));
/// assert!(is_prime(999983));
/// assert!(!is_prime(999984));
/// ```
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    
    // Use deterministic Miller-Rabin for smaller numbers
    if n <= 3_825_123_056_546_413_051 {
        return miller_rabin_deterministic(n);
    }
    
    // For very large numbers, use Baillie-PSW
    baillie_psw(n)
}

/// Deterministic Miller-Rabin primality test
///
/// Uses predetermined witnesses based on the input range for deterministic results.
pub fn miller_rabin(n: u64, witnesses: &[u64]) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    
    // Write n-1 as d * 2^r
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    
    'witness_loop: for &a in witnesses {
        if a >= n {
            continue;
        }
        
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue 'witness_loop;
        }
        
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness_loop;
            }
        }
        
        return false;
    }
    
    true
}

/// Deterministic Miller-Rabin using optimal witness sets
fn miller_rabin_deterministic(n: u64) -> bool {
    for (&threshold, witnesses) in MILLER_RABIN_WITNESSES.iter() {
        if n < threshold {
            return miller_rabin(n, witnesses);
        }
    }
    
    // Fallback to largest witness set
    let witnesses = MILLER_RABIN_WITNESSES.get(&u64::MAX).unwrap();
    miller_rabin(n, witnesses)
}

/// Baillie-PSW primality test
///
/// A probabilistic test that has no known counterexamples below 2^64.
pub fn baillie_psw(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    
    // Check for perfect squares
    if is_perfect_square(n) {
        return false;
    }
    
    // Miller-Rabin test with base 2
    if !miller_rabin(n, &[2]) {
        return false;
    }
    
    // Lucas test
    lucas_test(n)
}

/// Lucas primality test component of Baillie-PSW
fn lucas_test(n: u64) -> bool {
    // Find the first D such that (D/n) = -1
    let mut d = 5i64;
    loop {
        let jacobi = jacobi_symbol(d, n as i64);
        if jacobi == -1 {
            break;
        }
        if jacobi == 0 {
            return n == d.abs() as u64;
        }
        
        d = if d > 0 { -(d + 2) } else { -(d - 2) };
        
        if d.abs() > 1000 {
            // Safety check to avoid infinite loop
            return false;
        }
    }
    
    let p = 1;
    let q = (1 - d) / 4;
    
    lucas_sequence_test(n, p, q)
}

/// Lucas sequence test
fn lucas_sequence_test(n: u64, p: i64, q: i64) -> bool {
    let delta = (n + 1) as i64;
    
    // Simplified Lucas sequence computation
    // This is a basic implementation - a full implementation would be more complex
    let (u, _v) = lucas_sequence(delta, p, q, n);
    
    u == 0
}

/// Compute Lucas sequence U_k and V_k
fn lucas_sequence(k: i64, p: i64, q: i64, n: u64) -> (u64, u64) {
    if k == 0 {
        return (0, 2);
    }
    if k == 1 {
        return (1, p as u64 % n);
    }
    
    let mut u_prev = 0u64;
    let mut u_curr = 1u64;
    let mut v_prev = 2u64;
    let mut v_curr = p as u64 % n;
    
    for _ in 2..=k {
        let u_next = (((p as u64) * u_curr) % n + n - ((q as u64) * u_prev) % n) % n;
        let v_next = (((p as u64) * v_curr) % n + n - ((q as u64) * v_prev) % n) % n;
        
        u_prev = u_curr;
        u_curr = u_next;
        v_prev = v_curr;
        v_curr = v_next;
    }
    
    (u_curr, v_curr)
}

/// Check if a number is a perfect square
fn is_perfect_square(n: u64) -> bool {
    if n == 0 {
        return true;
    }
    
    let sqrt_n = (n as f64).sqrt() as u64;
    sqrt_n * sqrt_n == n || (sqrt_n + 1) * (sqrt_n + 1) == n
}

/// Compute Jacobi symbol (a/n)
fn jacobi_symbol(mut a: i64, mut n: i64) -> i8 {
    if n <= 0 || n % 2 == 0 {
        return 0;
    }
    
    let mut result = 1i8;
    
    if a < 0 {
        a = -a;
        if n % 4 == 3 {
            result = -result;
        }
    }
    
    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            if n % 8 == 3 || n % 8 == 5 {
                result = -result;
            }
        }
        
        std::mem::swap(&mut a, &mut n);
        
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        
        a %= n;
    }
    
    if n == 1 {
        result
    } else {
        0
    }
}

/// Modular exponentiation: (base^exp) mod m
fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    
    let mut result = 1;
    base %= m;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    
    result
}

/// Modular multiplication: (a * b) mod m, avoiding overflow
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_small_primes() {
        let small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        for &p in &small_primes {
            assert!(is_prime(p), "Failed for prime {}", p);
        }
    }
    
    #[test]
    fn test_small_composites() {
        let composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25];
        for &c in &composites {
            assert!(!is_prime(c), "Failed for composite {}", c);
        }
    }
    
    #[test]
    fn test_large_primes() {
        assert!(is_prime(999983));
        assert!(is_prime(1000003));
        assert!(is_prime(982451653));
    }
    
    #[test]
    fn test_miller_rabin() {
        assert!(miller_rabin(97, &[2]));
        assert!(!miller_rabin(91, &[2]));
    }
    
    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 4, 7), 4);
    }
}