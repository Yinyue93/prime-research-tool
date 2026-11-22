//! Arbitrary-precision primality testing and factorization.
//!
//! The rest of the library works on `u64`. This module handles inputs of any
//! size, delegating to the fast `u64` routines whenever a value fits.
//!
//! * Below [`DETERMINISTIC_BOUND`] (~3.3 × 10²⁴) primality is **proven**, using
//!   Miller-Rabin with the first 13 prime bases (Sorenson & Webster, 2015).
//! * Above it, Baillie-PSW (Miller-Rabin base 2 + strong Lucas) is used. No
//!   counterexample is known, but the result is a *probable* prime.
//! * Factorization uses trial division, perfect-power detection and
//!   Pollard-Brent rho, with an optional deadline because factoring large
//!   semiprimes can take arbitrarily long.

use crate::algorithms::is_prime as is_prime_u64;
use crate::factorization::factor as factor_u64;
use crate::sieve::sieve_range;
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use once_cell::sync::Lazy;
use rand::Rng;
use std::time::Instant;

/// Miller-Rabin with the first 13 primes as bases is deterministic below this value.
pub const DETERMINISTIC_BOUND: u128 = 3_317_044_064_679_887_385_961_981;

const DETERMINISTIC_BASES: [u64; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Primes used for trial division before the expensive tests.
static SMALL_PRIMES: Lazy<Vec<u64>> = Lazy::new(|| sieve_range(2..1000));

/// Result of a primality test on an arbitrary-precision integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Primality {
    /// Proven prime.
    Prime,
    /// Passed Baillie-PSW; no known counterexample, but not a proof.
    ProbablePrime,
    /// Proven composite.
    Composite,
}

impl Primality {
    /// True for both `Prime` and `ProbablePrime`.
    pub fn is_prime(self) -> bool {
        !matches!(self, Primality::Composite)
    }

    /// True when the answer is a proof (prime or composite), false for probable primes.
    pub fn is_proven(self) -> bool {
        !matches!(self, Primality::ProbablePrime)
    }
}

/// Test an arbitrary-precision integer for primality.
///
/// ```
/// use num_bigint::BigUint;
/// use prime_tool::bigint::{primality_big, Primality};
///
/// // 2^64 + 13 is the smallest prime above 2^64
/// let n: BigUint = "18446744073709551629".parse().unwrap();
/// assert_eq!(primality_big(&n), Primality::Prime);
/// ```
pub fn primality_big(n: &BigUint) -> Primality {
    if let Some(x) = n.to_u64() {
        return if is_prime_u64(x) {
            Primality::Prime
        } else {
            Primality::Composite
        };
    }

    // n > u64::MAX here, so n is never equal to one of the small primes.
    if SMALL_PRIMES.iter().any(|&p| (n % p).is_zero()) {
        return Primality::Composite;
    }

    if *n < BigUint::from(DETERMINISTIC_BOUND) {
        return if miller_rabin_big(n, &DETERMINISTIC_BASES) {
            Primality::Prime
        } else {
            Primality::Composite
        };
    }

    if !miller_rabin_big(n, &[2]) || is_perfect_square_big(n) || !strong_lucas_big(n) {
        Primality::Composite
    } else {
        Primality::ProbablePrime
    }
}

/// Result of factoring an arbitrary-precision integer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigFactorization {
    /// Prime factors found, in ascending order, with multiplicity.
    pub factors: Vec<BigUint>,
    /// Composite cofactors that could not be split before the deadline.
    /// The product of `factors` and `unfactored` always equals the input.
    pub unfactored: Vec<BigUint>,
    /// True if every entry in `factors` is proven prime; false if any is only a probable prime.
    pub proven: bool,
}

impl BigFactorization {
    /// True when the input was fully factored into primes.
    pub fn is_complete(&self) -> bool {
        self.unfactored.is_empty()
    }
}

/// Factor an arbitrary-precision integer.
///
/// `deadline` bounds the time spent in Pollard-Brent rho. Cofactors that are
/// still composite when it passes are returned in
/// [`BigFactorization::unfactored`]. `None` means no limit.
pub fn factor_big(n: &BigUint, deadline: Option<Instant>) -> BigFactorization {
    let mut result = BigFactorization {
        factors: Vec::new(),
        unfactored: Vec::new(),
        proven: true,
    };

    if let Some(x) = n.to_u64() {
        result.factors = factor_u64(x).into_iter().map(BigUint::from).collect();
        return result;
    }

    // Strip small prime factors.
    let mut rest = n.clone();
    for &p in SMALL_PRIMES.iter() {
        while (&rest % p).is_zero() {
            rest /= p;
            result.factors.push(BigUint::from(p));
        }
    }

    let mut stack = vec![rest];
    while let Some(m) = stack.pop() {
        if m.is_one() {
            continue;
        }
        if let Some(x) = m.to_u64() {
            result
                .factors
                .extend(factor_u64(x).into_iter().map(BigUint::from));
            continue;
        }

        let primality = primality_big(&m);
        if primality.is_prime() {
            result.proven &= primality.is_proven();
            result.factors.push(m);
            continue;
        }

        if let Some((root, k)) = perfect_power_big(&m) {
            stack.extend(std::iter::repeat_n(root, k as usize));
            continue;
        }

        match brent_big(&m, deadline) {
            Some(d) => {
                let cofactor = &m / &d;
                stack.push(d);
                stack.push(cofactor);
            }
            None => result.unfactored.push(m),
        }
    }

    result.factors.sort();
    result.unfactored.sort();
    result
}

/// Miller-Rabin with the given bases. Requires odd `n` larger than every base.
fn miller_rabin_big(n: &BigUint, bases: &[u64]) -> bool {
    let one = BigUint::one();
    let n_minus_1 = n - &one;
    let s = n_minus_1.trailing_zeros().expect("n - 1 is nonzero");
    let d = &n_minus_1 >> s;

    'witness: for &a in bases {
        let mut x = BigUint::from(a).modpow(&d, n);
        if x == one || x == n_minus_1 {
            continue;
        }
        for _ in 1..s {
            x = &x * &x % n;
            if x == n_minus_1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Strong Lucas probable-prime test with Selfridge parameters (P = 1).
/// Requires odd `n` that is not a perfect square and has no small factors.
fn strong_lucas_big(n: &BigUint) -> bool {
    let mut d: i64 = 5;
    loop {
        match jacobi_big(d, n) {
            -1 => break,
            0 => return false, // gcd(d, n) > 1, and n > |d|, so n is composite
            _ => {}
        }
        d = if d > 0 { -(d + 2) } else { -(d - 2) };
    }

    let q = to_mod_big((1 - d) / 4, n);
    let d_mod = to_mod_big(d, n);

    // n + 1 = k * 2^s with k odd
    let n_plus_1 = n + 1u32;
    let s = n_plus_1.trailing_zeros().expect("n + 1 is nonzero");
    let k = &n_plus_1 >> s;

    // U_1 = 1, V_1 = P = 1, Q^1
    let mut u = BigUint::one();
    let mut v = BigUint::one();
    let mut qk = q.clone();

    for bit in (0..k.bits() - 1).rev() {
        // Doubling: U_2m = U_m V_m, V_2m = V_m^2 - 2 Q^m, Q^2m = (Q^m)^2
        u = &u * &v % n;
        v = sub_mod(&(&v * &v % n), &((&qk << 1u32) % n), n);
        qk = &qk * &qk % n;
        if k.bit(bit) {
            // Increment: U_m+1 = (P U + V) / 2, V_m+1 = (D U + P V) / 2
            let u_next = half_mod((&u + &v) % n, n);
            let v_next = half_mod((&d_mod * &u + &v) % n, n);
            u = u_next;
            v = v_next;
            qk = &qk * &q % n;
        }
    }

    if u.is_zero() || v.is_zero() {
        return true;
    }
    for _ in 1..s {
        v = sub_mod(&(&v * &v % n), &((&qk << 1u32) % n), n);
        if v.is_zero() {
            return true;
        }
        qk = &qk * &qk % n;
    }
    false
}

/// Jacobi symbol (a/n) for odd n > 0.
fn jacobi_big(a: i64, n: &BigUint) -> i8 {
    let mut a = to_mod_big(a, n);
    let mut n = n.clone();
    let mut result = 1i8;

    while !a.is_zero() {
        let tz = a.trailing_zeros().expect("a is nonzero");
        a >>= tz;
        let n_mod_8 = (&n % 8u32).to_u32().expect("small");
        if tz % 2 == 1 && (n_mod_8 == 3 || n_mod_8 == 5) {
            result = -result;
        }
        std::mem::swap(&mut a, &mut n);
        if (&a % 4u32).to_u32() == Some(3) && (&n % 4u32).to_u32() == Some(3) {
            result = -result;
        }
        a = &a % &n;
    }

    if n.is_one() {
        result
    } else {
        0
    }
}

/// Pollard-Brent rho. Returns a nontrivial factor of composite `n`,
/// or `None` if `deadline` passes first.
fn brent_big(n: &BigUint, deadline: Option<Instant>) -> Option<BigUint> {
    const BATCH: u64 = 128;
    let one = BigUint::one();
    let expired = || deadline.is_some_and(|d| Instant::now() >= d);
    let mut rng = rand::thread_rng();
    let abs_diff = |a: &BigUint, b: &BigUint| if a > b { a - b } else { b - a };

    loop {
        if expired() {
            return None;
        }

        let c = BigUint::from(rng.gen_range(1..u64::MAX)) % n;
        let f = |x: &BigUint| (x * x + &c) % n;

        let mut y = BigUint::from(rng.gen_range(2..u64::MAX)) % n;
        let mut x = y.clone();
        let mut ys = y.clone();
        let mut q = one.clone();
        let mut g = one.clone();
        let mut r: u64 = 1;

        while g == one {
            x = y.clone();
            for _ in 0..r {
                y = f(&y);
            }
            let mut k = 0;
            while k < r && g == one {
                ys = y.clone();
                for _ in 0..BATCH.min(r - k) {
                    y = f(&y);
                    q = q * abs_diff(&x, &y) % n;
                }
                g = q.gcd(n);
                k += BATCH;
                if expired() {
                    return None;
                }
            }
            r *= 2;
        }

        if g == *n {
            // The batched product overshot; step back one at a time.
            loop {
                ys = f(&ys);
                g = abs_diff(&x, &ys).gcd(n);
                if g != one {
                    break;
                }
            }
        }

        if g != *n {
            return Some(g);
        }
        // Cycle closed without splitting n; retry with a different polynomial.
    }
}

/// If n = r^k for some k ≥ 2, return (r, k) with the largest such k.
fn perfect_power_big(n: &BigUint) -> Option<(BigUint, u32)> {
    let max_k = n.bits() as u32;
    for k in (2..=max_k).rev() {
        let r = n.nth_root(k);
        if r > BigUint::one() && r.pow(k) == *n {
            return Some((r, k));
        }
    }
    None
}

fn is_perfect_square_big(n: &BigUint) -> bool {
    let r = n.sqrt();
    &r * &r == *n
}

/// Reduce a signed value into [0, n).
fn to_mod_big(x: i64, n: &BigUint) -> BigUint {
    let m = BigUint::from(x.unsigned_abs()) % n;
    if x >= 0 || m.is_zero() {
        m
    } else {
        n - m
    }
}

/// (a - b) mod n for a, b in [0, n).
fn sub_mod(a: &BigUint, b: &BigUint, n: &BigUint) -> BigUint {
    if a >= b {
        a - b
    } else {
        n - b + a
    }
}

/// x / 2 mod n for odd n and x in [0, n).
fn half_mod(x: BigUint, n: &BigUint) -> BigUint {
    if x.is_even() {
        x >> 1u32
    } else {
        (x + n) >> 1u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(s: &str) -> BigUint {
        s.parse().unwrap()
    }

    fn product(f: &BigFactorization) -> BigUint {
        f.factors
            .iter()
            .chain(&f.unfactored)
            .fold(BigUint::one(), |acc, x| acc * x)
    }

    #[test]
    fn agrees_with_u64_path() {
        for n in 0..5_000u64 {
            assert_eq!(
                primality_big(&BigUint::from(n)).is_prime(),
                is_prime_u64(n),
                "n = {}",
                n
            );
        }
    }

    #[test]
    fn primes_just_above_u64() {
        // 2^64 + 13 is the smallest prime above 2^64: proven, below the deterministic bound
        assert_eq!(
            primality_big(&big("18446744073709551629")),
            Primality::Prime
        );
        for k in 0..13u32 {
            let n = (BigUint::one() << 64u32) + k;
            assert_eq!(primality_big(&n), Primality::Composite, "2^64 + {}", k);
        }
    }

    #[test]
    fn mersenne_primes_above_deterministic_bound() {
        for p in [89u32, 107, 127, 521] {
            let m = (BigUint::one() << p) - 1u32;
            assert_eq!(primality_big(&m), Primality::ProbablePrime, "2^{} - 1", p);
        }
        // 2^67 - 1 = 193707721 × 761838257287 (Cole, 1903)
        let m67 = (BigUint::one() << 67u32) - 1u32;
        assert_eq!(primality_big(&m67), Primality::Composite);
    }

    #[test]
    fn rejects_strong_pseudoprimes() {
        // Strong pseudoprime to all prime bases up to 37 (Zhang & Tang),
        // above u64 but below the deterministic bound: base 41 must catch it.
        let n = big("318665857834031151167461");
        assert_eq!(primality_big(&n), Primality::Composite);
        // Product of two large primes above the deterministic bound
        let n = big("2305843009213693951") * big("618970019642690137449562111");
        assert_eq!(primality_big(&n), Primality::Composite);
    }

    #[test]
    fn factors_mersenne_67() {
        let n = (BigUint::one() << 67u32) - 1u32;
        let f = factor_big(&n, None);
        assert!(f.is_complete() && f.proven);
        assert_eq!(f.factors, vec![big("193707721"), big("761838257287")]);
    }

    #[test]
    fn factors_mixed_number() {
        // 2^5 × 3^2 × 999999937 × 1000000007 × (2^89 - 1).
        // Rho cost is ~sqrt(smallest remaining factor), so keep all but one
        // large factor around 10^9 to keep this test fast.
        let m89 = big("618970019642690137449562111");
        let n = BigUint::from(32u32 * 9) * 999_999_937u64 * 1_000_000_007u64 * &m89;
        let f = factor_big(&n, None);
        assert!(f.is_complete());
        assert!(!f.proven, "2^89 - 1 is only a probable prime");
        assert_eq!(product(&f), n);
        assert_eq!(
            f.factors
                .iter()
                .filter(|x| **x == BigUint::from(2u32))
                .count(),
            5
        );
        assert!(f.factors.contains(&m89));
        assert!(f.factors.contains(&BigUint::from(999_999_937u64)));
    }

    #[test]
    fn factors_perfect_power() {
        let p = big("18446744073709551629"); // 2^64 + 13
        let n = p.pow(3);
        let f = factor_big(&n, None);
        assert_eq!(f.factors, vec![p.clone(), p.clone(), p]);
        assert!(f.proven);
    }

    #[test]
    fn deadline_leaves_hard_cofactor_unfactored() {
        // Two ~30-digit primes: far out of reach for rho in this time budget
        let n = big("618970019642690137449562111") * big("162259276829213363391578010288127");
        let f = factor_big(&n, Some(Instant::now()));
        assert!(!f.is_complete());
        assert_eq!(product(&f), n);
    }
}
