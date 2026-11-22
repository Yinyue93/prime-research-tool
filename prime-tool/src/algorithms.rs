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
    
    // Deterministic Miller-Rabin covers the entire u64 range: the largest
    // witness set (first 12 primes) is proven correct for all n < 3.3e24.
    miller_rabin_deterministic(n)
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

/// Strong Lucas probable-prime test (Selfridge method A), the second half
/// of Baillie-PSW. Runs in O(log n) using the doubling formulas, with all
/// arithmetic done mod n in u128 to avoid overflow across the whole u64 range.
fn lucas_test(n: u64) -> bool {
    // Find the first D in 5, -7, 9, -11, ... with Jacobi symbol (D/n) = -1.
    // Callers have already rejected perfect squares, so this terminates quickly.
    let mut d: i64 = 5;
    loop {
        match jacobi_symbol(d, n) {
            -1 => break,
            0 if d.unsigned_abs() != n => return false, // nontrivial factor of n
            _ => {}
        }
        d = if d > 0 { -(d + 2) } else { -(d - 2) };
    }

    let p: u64 = 1;
    let q_mod = to_mod((1 - d) / 4, n);
    let d_mod = to_mod(d, n);

    // Write n + 1 = k * 2^s with k odd (n is odd and < 2^64, so use u128).
    let np1 = n as u128 + 1;
    let s = np1.trailing_zeros();
    let k = np1 >> s;

    // Compute U_k, V_k, Q^k mod n, scanning bits of k from the top.
    let (mut u, mut v, mut qk) = (1u64, p % n, q_mod); // U_1, V_1, Q^1
    for bit in (0..(128 - k.leading_zeros() - 1)).rev() {
        // Doubling: U_2m = U_m V_m, V_2m = V_m^2 - 2 Q^m, Q^2m = (Q^m)^2
        u = mod_mul(u, v, n);
        v = mod_sub(mod_mul(v, v, n), mod_mul(2, qk, n), n);
        qk = mod_mul(qk, qk, n);
        if (k >> bit) & 1 == 1 {
            // Increment: U_m+1 = (P U + V)/2, V_m+1 = (D U + P V)/2
            let u_next = half_mod(mod_add(mod_mul(p, u, n), v, n), n);
            let v_next = half_mod(mod_add(mod_mul(d_mod, u, n), mod_mul(p, v, n), n), n);
            u = u_next;
            v = v_next;
            qk = mod_mul(qk, q_mod, n);
        }
    }

    if u == 0 || v == 0 {
        return true;
    }
    for _ in 1..s {
        v = mod_sub(mod_mul(v, v, n), mod_mul(2, qk, n), n);
        if v == 0 {
            return true;
        }
        qk = mod_mul(qk, qk, n);
    }
    false
}

/// Reduce a signed value into [0, n)
fn to_mod(x: i64, n: u64) -> u64 {
    (x as i128).rem_euclid(n as i128) as u64
}

fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + b as u128) % m as u128) as u64
}

fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + m as u128 - b as u128) % m as u128) as u64
}

/// x / 2 mod m for odd m
fn half_mod(x: u64, m: u64) -> u64 {
    if x % 2 == 0 {
        x / 2
    } else {
        ((x as u128 + m as u128) / 2) as u64
    }
}

/// Check if a number is a perfect square (exact for all u64)
fn is_perfect_square(n: u64) -> bool {
    let mut r = (n as f64).sqrt() as u64;
    // Correct floating-point error in either direction
    while (r as u128) * (r as u128) > n as u128 {
        r -= 1;
    }
    while ((r + 1) as u128) * ((r + 1) as u128) <= n as u128 {
        r += 1;
    }
    (r as u128) * (r as u128) == n as u128
}

/// Compute the Jacobi symbol (a/n) for odd n > 0
fn jacobi_symbol(a: i64, n: u64) -> i8 {
    if n == 0 || n % 2 == 0 {
        return 0;
    }
    let mut a = to_mod(a, n);
    let mut n = n;
    let mut result = 1i8;

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

    if n == 1 { result } else { 0 }
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
    fn test_primes_near_u64_max() {
        // Above the old Baillie-PSW cutoff (3.8e18) and above 2^63
        assert!(is_prime(9_223_372_036_854_775_783)); // largest prime < 2^63
        assert!(is_prime(18_446_744_073_709_551_557)); // 2^64 - 59, largest u64 prime
        assert!(is_prime(18_446_744_073_709_551_533));
        assert!(!is_prime(18_446_744_073_709_551_559));
        assert!(!is_prime(u64::MAX));
        // 4294967291 * 4294967279: product of two large primes
        assert!(!is_prime(4_294_967_291 * 4_294_967_279));
    }
    
    #[test]
    fn test_miller_rabin() {
        assert!(miller_rabin(97, &[2]));
        assert!(!miller_rabin(91, &[2]));
    }
    
    #[test]
    fn test_baillie_psw_matches_miller_rabin() {
        // Small range, exhaustively
        for n in 0..20_000u64 {
            assert_eq!(baillie_psw(n), is_prime(n), "mismatch at {}", n);
        }
        // Strong pseudoprimes to base 2 and Carmichael numbers must be rejected
        for &c in &[2047u64, 3277, 4033, 561, 1105, 1729, 3_215_031_751, 3_825_123_056_546_413_051] {
            assert!(!baillie_psw(c), "{} is composite", c);
        }
        // Large values near u64::MAX
        for n in (u64::MAX - 2000)..=u64::MAX {
            assert_eq!(baillie_psw(n), is_prime(n), "mismatch at {}", n);
        }
    }
    
    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 4, 7), 4);
    }
}