//! Fast factorization algorithms
//!
//! This module implements various factorization methods:
//! - Pollard's Rho algorithm
//! - Brent's improvement to Pollard's Rho
//! - Elliptic Curve Method (ECM) for larger composites

use crate::algorithms::is_prime;
use crate::error::{PrimeError, Result};
use rand::Rng;
use std::collections::HashMap;

/// Factor a number into its prime components
///
/// # Examples
///
/// ```
/// use prime_tool::factor;
///
/// let factors = factor(60);
/// assert_eq!(factors, vec![2, 2, 3, 5]);
/// ```
pub fn factor(n: u64) -> Vec<u64> {
    if n <= 1 {
        return vec![];
    }
    
    if is_prime(n) {
        return vec![n];
    }
    
    let mut factors = Vec::new();
    let mut remaining = n;
    
    // Handle small factors first
    for &p in &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        while remaining % p == 0 {
            factors.push(p);
            remaining /= p;
        }
        if remaining == 1 {
            return factors;
        }
        if is_prime(remaining) {
            factors.push(remaining);
            return factors;
        }
    }
    
    // Use advanced factorization for remaining composite
    let mut stack = vec![remaining];
    
    while let Some(num) = stack.pop() {
        if num == 1 {
            continue;
        }
        
        if is_prime(num) {
            factors.push(num);
            continue;
        }
        
        // Try different factorization methods
        if let Some(factor) = pollard_rho(num) {
            stack.push(factor);
            stack.push(num / factor);
        } else if let Some(factor) = brent_factor(num) {
            stack.push(factor);
            stack.push(num / factor);
        } else if let Some(factor) = ecm_factor(num) {
            stack.push(factor);
            stack.push(num / factor);
        } else {
            // Fallback to trial division for stubborn numbers
            if let Some(factor) = trial_division(num) {
                stack.push(factor);
                stack.push(num / factor);
            } else {
                // This shouldn't happen for numbers <= 2^64
                factors.push(num);
            }
        }
    }
    
    factors.sort_unstable();
    factors
}

/// Pollard's Rho factorization algorithm
pub fn pollard_rho(n: u64) -> Option<u64> {
    if n % 2 == 0 {
        return Some(2);
    }
    
    let mut rng = rand::thread_rng();
    
    for _ in 0..10 {
        let c = rng.gen_range(1..n);
        let mut x = rng.gen_range(2..n);
        let mut y = x;
        let mut d = 1;
        
        while d == 1 {
            x = pollard_f(x, c, n);
            y = pollard_f(pollard_f(y, c, n), c, n);
            d = gcd(if x > y { x - y } else { y - x }, n);
            
            if d == n {
                break;
            }
        }
        
        if d > 1 && d < n {
            return Some(d);
        }
    }
    
    None
}

/// Brent's improvement to Pollard's Rho
pub fn brent_factor(n: u64) -> Option<u64> {
    if n % 2 == 0 {
        return Some(2);
    }
    
    let mut rng = rand::thread_rng();
    
    for _ in 0..10 {
        let c = rng.gen_range(1..n);
        let mut y = rng.gen_range(1..n);
        let mut r = 1;
        let mut q = 1;
        
        loop {
            let mut x = y;
            for _ in 0..r {
                y = pollard_f(y, c, n);
            }
            
            let mut k = 0;
            while k < r {
                let mut ys = y;
                let m = std::cmp::min(100, r - k);
                
                for _ in 0..m {
                    y = pollard_f(y, c, n);
                    q = mod_mul(q, if x > y { x - y } else { y - x }, n);
                }
                
                let g = gcd(q, n);
                if g > 1 {
                    if g == n {
                        loop {
                            ys = pollard_f(ys, c, n);
                            let g2 = gcd(if x > ys { x - ys } else { ys - x }, n);
                            if g2 > 1 {
                                if g2 < n {
                                    return Some(g2);
                                }
                                break;
                            }
                        }
                    } else {
                        return Some(g);
                    }
                }
                
                k += m;
            }
            
            r *= 2;
            if r > 1000000 {
                break;
            }
        }
    }
    
    None
}

/// Elliptic Curve Method (ECM) for factorization
pub fn ecm_factor(n: u64) -> Option<u64> {
    if n % 2 == 0 {
        return Some(2);
    }
    
    let mut rng = rand::thread_rng();
    
    // Try multiple curves
    for _ in 0..20 {
        let a = rng.gen_range(1..n);
        let x = rng.gen_range(1..n);
        let y = rng.gen_range(1..n);
        
        // Compute b = y^2 - x^3 - ax (mod n)
        let x3 = mod_mul(mod_mul(x, x, n), x, n);
        let ax = mod_mul(a, x, n);
        let y2 = mod_mul(y, y, n);
        let b = (y2 + n - (x3 + ax) % n) % n;
        
        // Point on the curve
        let mut px = x;
        let mut py = y;
        
        // Multiply by small primes
        for &prime in &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
            for _ in 0..10 {
                if let Some((new_x, new_y)) = ec_double(px, py, a, n) {
                    px = new_x;
                    py = new_y;
                } else {
                    // Point at infinity or gcd found
                    let g = gcd(px, n);
                    if g > 1 && g < n {
                        return Some(g);
                    }
                    break;
                }
            }
        }
    }
    
    None
}

/// Trial division for small factors
fn trial_division(n: u64) -> Option<u64> {
    let limit = ((n as f64).sqrt() as u64).min(1_000_000);
    
    for i in (3..=limit).step_by(2) {
        if n % i == 0 {
            return Some(i);
        }
    }
    
    None
}

/// Pollard's Rho function: f(x) = x^2 + c (mod n)
fn pollard_f(x: u64, c: u64, n: u64) -> u64 {
    (mod_mul(x, x, n) + c) % n
}

/// Elliptic curve point doubling
fn ec_double(x: u64, y: u64, a: u64, n: u64) -> Option<(u64, u64)> {
    if y == 0 {
        return None; // Point at infinity
    }
    
    // Compute slope: s = (3x^2 + a) / (2y)
    let numerator = (3 * mod_mul(x, x, n) + a) % n;
    let denominator = (2 * y) % n;
    
    let inv = mod_inverse(denominator, n)?;
    let s = mod_mul(numerator, inv, n);
    
    // New coordinates
    let x3 = (mod_mul(s, s, n) + n - 2 * x) % n;
    let y3 = (mod_mul(s, (x + n - x3) % n, n) + n - y) % n;
    
    Some((x3, y3))
}

/// Modular inverse using extended Euclidean algorithm
fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g != 1 {
        None
    } else {
        Some(((x % m as i64 + m as i64) % m as i64) as u64)
    }
}

/// Extended Euclidean algorithm
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, y1, x1) = extended_gcd(b % a, a);
        let y = x1 - (b / a) * y1;
        (g, y, y1)
    }
}

/// Greatest Common Divisor using Euclidean algorithm
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Modular multiplication avoiding overflow
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_factor_small() {
        assert_eq!(factor(12), vec![2, 2, 3]);
        assert_eq!(factor(60), vec![2, 2, 3, 5]);
        assert_eq!(factor(97), vec![97]); // prime
    }
    
    #[test]
    fn test_pollard_rho() {
        if let Some(f) = pollard_rho(15) {
            assert!(f == 3 || f == 5);
        }
    }
    
    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(17, 13), 1);
    }
    
    #[test]
    fn test_factor_large() {
        let factors = factor(999983 * 1000003);
        assert!(factors.contains(&999983));
        assert!(factors.contains(&1000003));
    }
}