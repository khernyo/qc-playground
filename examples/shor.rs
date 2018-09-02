extern crate num;
extern crate num_bigint;
extern crate rand;

extern crate qc_playground;

use std::ops::Sub;

use num::pow::pow;
use num::{Integer, Num, One, ToPrimitive};
use num_bigint::BigInt;
use rand::{thread_rng, Rng};

use qc_playground::Qubit;

fn main() {
    println!("{:?}", shor(5, cpf_naive));
    println!("{:?}", shor(15, qpf));
}

#[derive(Clone, Debug)]
enum ShorResult {
    None,
    Single(u32),
    Double(u32, u32),
}

fn shor<F>(n: u32, pf: F) -> ShorResult
where
    F: Fn(&BigInt, &BigInt) -> u32,
{
    println!("n = {}", n);
    assert!(n.is_odd());
    assert!(n > 3);
    assert!(n < 50);
    let n_bi: BigInt = n.into();
    let mut rng = thread_rng();
    for i in 0..n * 2 {
        let a = rng.gen_range(3, n);
        let a: BigInt = a.into();
        println!("a = {}", a);
        let a_gcd_n = a.gcd(&n_bi);
        println!("gcd(a, n) = {}", a_gcd_n);
        if a_gcd_n != BigInt::one() {
            return ShorResult::Single(a_gcd_n.to_u32().unwrap());
        } else {
            let r = pf(&a, &n_bi);
            println!("r = {}", r);
            if r.is_odd() {
                println!("r is odd");
                continue;
            } else {
                let p = pow(a, (r / 2) as usize);
                if is_congruent(&p, -1, &n_bi) {
                    println!("a^(r/2) ≡ -1 (mod n)");
                    continue;
                } else {
                    return ShorResult::Double(
                        (&p + BigInt::one()).gcd(&n_bi).to_u32().unwrap(),
                        (&p - BigInt::one()).gcd(&n_bi).to_u32().unwrap(),
                    );
                }
            }
        }
    }
    ShorResult::None
}

/// a ≡ b (mod n)
fn is_congruent(a: &BigInt, b: i32, n: &BigInt) -> bool {
    (a - BigInt::from(b)).is_multiple_of(n)
}

/// Quantum period-finding
///
/// Returns `r`, the period of `f(x) = a^x mod n`.
fn qpf(a: &BigInt, n: &BigInt) -> u32 {
    unimplemented!()
}

/// Classical period-finding
///
/// Returns `r`, the period of `f(x) = a^x mod n`.
fn cpf_naive(a: &BigInt, n: &BigInt) -> u32 {
    const MAX: u32 = 1000;
    let fv = |r| {
        (0..r * 2)
            .map(|x| a.modpow(&x.into(), &n).to_u32().unwrap())
            .collect::<Vec<_>>()
    };
    for r in 1..MAX {
        let v = fv(r);
        if check_period(&v, r) {
            println!("period: {} {:?}", r, v);
            return r;
        }
    }
    panic!("cpf_naive: a={} n={} v={:?}", a, n, fv(MAX));
}

fn check_period(v: &[u32], r: u32) -> bool {
    assert_ne!(r, 0);
    assert!(r <= v.len() as u32);
    let (i, j) = (0, r);
    v.iter().zip(v.iter().skip(r as usize)).all(|(x, y)| x == y)
}

#[cfg(test)]
mod tests {
    #![feature(step_by)]

    extern crate is_prime;

    use self::is_prime::is_prime;

    use super::*;

    #[test]
    #[should_panic]
    fn test_check_period_panic1() {
        check_period(&vec![0], 0);
    }

    #[test]
    #[should_panic]
    fn test_check_period_panic2() {
        check_period(&vec![], 1);
    }

    #[test]
    fn test_check_period() {
        assert_eq!(check_period(&vec![0], 1), true);
        assert_eq!(check_period(&vec![0, 0, 0], 1), true);
        assert_eq!(check_period(&vec![0, 0, 0], 2), true);
        assert_eq!(check_period(&vec![0, 0, 0], 3), true);
        assert_eq!(check_period(&vec![0, 1, 2], 1), false);
        assert_eq!(check_period(&vec![0, 1, 2], 2), false);
        assert_eq!(check_period(&vec![0, 1, 2], 3), true);
        assert_eq!(check_period(&vec![0, 1, 0], 1), false);
        assert_eq!(check_period(&vec![0, 1, 0], 2), true);
        assert_eq!(check_period(&vec![0, 1, 0, 1], 1), false);
        assert_eq!(check_period(&vec![0, 1, 0, 1], 2), true);
        assert_eq!(check_period(&vec![0, 1, 0, 1], 3), false);
        assert_eq!(check_period(&vec![0, 1, 0, 1], 4), true);
        assert_eq!(check_period(&vec![0, 1, 3, 1], 1), false);
        assert_eq!(check_period(&vec![0, 1, 3, 1], 2), false);
        assert_eq!(check_period(&vec![0, 1, 3, 1], 3), false);
        assert_eq!(check_period(&vec![0, 1, 3, 1], 4), true);
        assert_eq!(check_period(&vec![0, 1, 3, 0], 1), false);
        assert_eq!(check_period(&vec![0, 1, 3, 0], 2), false);
        assert_eq!(check_period(&vec![0, 1, 3, 0], 3), true);
        assert_eq!(check_period(&vec![0, 1, 3, 0], 4), true);
        assert_eq!(check_period(&vec![0, 1, 3, 0, 1, 3], 1), false);
        assert_eq!(check_period(&vec![0, 1, 3, 0, 1, 3], 2), false);
        assert_eq!(check_period(&vec![0, 1, 3, 0, 1, 3], 3), true);
        assert_eq!(check_period(&vec![0, 1, 3, 0, 1, 3], 4), false);
        assert_eq!(check_period(&vec![0, 1, 3, 0, 1, 3], 5), false);
        assert_eq!(check_period(&vec![0, 1, 3, 0, 1, 3], 6), true);
    }

    #[test]
    fn test_shor_cpf_naive() {
        for n in (5..40).step_by(2) {
            match shor(n, cpf_naive) {
                ShorResult::None => assert!(is_prime(&format!("{}", n))),
                ShorResult::Single(a) => assert_eq!(n % a, 0),
                ShorResult::Double(a, b) => {
                    assert_eq!(n % a, 0);
                    assert_eq!(n % b, 0);
                }
            }
        }
    }
}
