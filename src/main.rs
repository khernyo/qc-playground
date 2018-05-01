#[macro_use]
extern crate approx;
extern crate nalgebra;
extern crate num_complex;
extern crate num_traits;
extern crate rand;

use std::fmt::{self, Debug, Display, Formatter};

use approx::ApproxEq;
use nalgebra::*;
use num_complex::Complex64;
use rand::Rng;

fn main() {
    println!("{}", Qubit::PLUS);
}

#[derive(Clone)]
struct Qubit {
    a: Complex64,
    b: Complex64,
}

impl Qubit {
    /// |0>
    pub const ZERO: Qubit = Qubit {
        a: Complex64 { re: 1f64, im: 0f64 },
        b: Complex64 { re: 0f64, im: 0f64 },
    };

    /// |1>
    pub const ONE: Qubit = Qubit {
        a: Complex64 { re: 0f64, im: 0f64 },
        b: Complex64 { re: 1f64, im: 0f64 },
    };

    /// |+>
    pub const PLUS: Qubit = Qubit {
        a: Complex64 {
            re: std::f64::consts::FRAC_1_SQRT_2,
            im: 0f64,
        },
        b: Complex64 {
            re: std::f64::consts::FRAC_1_SQRT_2,
            im: 0f64,
        },
    };

    /// |->
    pub const MINUS: Qubit = Qubit {
        a: Complex64 {
            re: std::f64::consts::FRAC_1_SQRT_2,
            im: 0f64,
        },
        b: Complex64 {
            re: -std::f64::consts::FRAC_1_SQRT_2,
            im: 0f64,
        },
    };

    pub fn new() -> Qubit {
        Qubit::ZERO
    }

    pub fn from_components(a: Complex64, b: Complex64) -> Qubit {
        let q = Qubit::from_components_unchecked(a, b);
        q.validate();
        q
    }

    fn from_components_unchecked(a: Complex64, b: Complex64) -> Qubit {
        Qubit { a, b }
    }

    fn validate(&self) {
        self.prob();
    }

    fn prob(&self) -> (f64, f64) {
        let (zp, op) = (self.a.norm_sqr(), self.b.norm_sqr());
        assert_relative_eq!(zp + op, 1f64);
        (zp, op)
    }
}

impl Debug for Qubit {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "Qubit({}, {})", self.a, self.b)
    }
}

impl Display for Qubit {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        let (zp, op) = self.prob();
        if relative_eq!(zp, 1f64) {
            write!(f, "|0>")
        } else if relative_eq!(op, 1f64) {
            write!(f, "|1>")
        } else {
            write!(f, "{}|0> + {}|1>", self.a, self.b)
        }
    }
}

#[derive(Debug)]
struct QState {
    state: DVector<Complex64>,
}

impl QState {
    fn from_independent_qubits(qubits: &[Qubit]) -> QState {
        assert!(!qubits.is_empty());
        let state_len = 1usize << qubits.len();
        let qstate = {
            fn probability_of_state(state_idx: usize, qubits: &[Qubit]) -> Complex64 {
                let mut p = Complex64::new(1f64, 0f64);
                for i in 0..qubits.len() {
                    let b = state_idx >> i & 1;
                    let qubit = &qubits[qubits.len() - i - 1];
                    let q = if b == 0 { qubit.a } else { qubit.b };
                    p *= q;
                }
                p
            }
            let r: Vec<_> = (0..state_len)
                .map(|i| probability_of_state(i, &qubits))
                .collect();
            QState {
                state: DVector::from_row_slice(state_len, &r),
            }
        };
        assert_eq!(qstate.qubit_count() as usize, qubits.len());
        qstate.validate();
        qstate
    }

    fn validate(&self) {
        assert_relative_eq!(self.state.iter().map(|c| c.norm_sqr()).sum(), 1f64)
    }

    fn qubit_count(&self) -> u32 {
        assert!(self.state.len().is_power_of_two());
        self.state.len().trailing_zeros()
    }

    fn apply(self, gate: &QGate1) -> QState {
        fn to_dynamic(m: &Matrix2<Complex64>) -> DMatrix<Complex64> {
            m.clone().resize(m.nrows(), m.ncols(), Complex64::default())
        }
        QState {
            state: to_dynamic(&gate.0) * self.state,
        }
    }

    fn apply2(self, gate: &QGate2) -> QState {
        fn to_dynamic(m: &Matrix4<Complex64>) -> DMatrix<Complex64> {
            m.clone().resize(m.nrows(), m.ncols(), Complex64::default())
        }
        QState {
            state: to_dynamic(&gate.0) * self.state,
        }
    }

    fn measure(self, rng: &mut Rng) -> Vec<bool> {
        let p: Vec<_> = self.state.iter().map(|c| c.norm_sqr()).collect();
        let v = rng.next_f64();
        let mut p_acc = 0f64;
        for (i, item) in p.iter().enumerate() {
            p_acc += item;
            if v < p_acc {
                return n_to_bitvec(i, self.qubit_count());
            }
        }
        panic!();
    }
}

fn n_to_bitvec(n: usize, bits: u32) -> Vec<bool> {
    assert!(bits > 0);
    assert!(
        1usize << bits > n,
        "Cannot represent {} with {} bits",
        n,
        bits
    );
    (0..bits).rev().map(|i| n >> i & 1 == 1).collect()
}

#[derive(Debug)]
struct QGate1(Matrix2<Complex64>);

impl QGate1 {
    fn hadamard() -> QGate1 {
        let m: Matrix2<Complex64> = Matrix2::from_rows(&[
            RowVector2::new(1f64.into(), 1f64.into()),
            RowVector2::new(1f64.into(), (-1f64).into()),
        ]);
        QGate1(m / Complex64::new(2f64.sqrt(), 0f64))
    }

    fn pauli_x() -> QGate1 {
        QGate1(Matrix2::from_rows(&[
            RowVector2::new(0f64.into(), 1f64.into()),
            RowVector2::new(1f64.into(), 0f64.into()),
        ]))
    }

    fn pauli_y() -> QGate1 {
        QGate1(Matrix2::from_rows(&[
            RowVector2::new(0f64.into(), -Complex64::i()),
            RowVector2::new(Complex64::i(), 0f64.into()),
        ]))
    }

    fn pauli_z() -> QGate1 {
        QGate1(Matrix2::from_rows(&[
            RowVector2::new(1f64.into(), 0f64.into()),
            RowVector2::new(0f64.into(), (-1f64).into()),
        ]))
    }

    fn phase_shift(phi: f64) -> QGate1 {
        QGate1(Matrix2::from_rows(&[
            RowVector2::new(1f64.into(), 0f64.into()),
            RowVector2::new(0f64.into(), Complex64::from_polar(&1f64, &phi)),
        ]))
    }
}

#[derive(Debug)]
struct QGate2(Matrix4<Complex64>);

impl QGate2 {
    fn swap() -> QGate2 {
        QGate2(Matrix4::from_rows(&[
            RowVector4::new(1f64.into(), 0f64.into(), 0f64.into(), 0f64.into()),
            RowVector4::new(0f64.into(), 0f64.into(), 1f64.into(), 0f64.into()),
            RowVector4::new(0f64.into(), 1f64.into(), 0f64.into(), 0f64.into()),
            RowVector4::new(0f64.into(), 0f64.into(), 0f64.into(), 1f64.into()),
        ]))
    }

    fn cnot() -> QGate2 {
        QGate2(Matrix4::from_rows(&[
            RowVector4::new(1f64.into(), 0f64.into(), 0f64.into(), 0f64.into()),
            RowVector4::new(0f64.into(), 1f64.into(), 0f64.into(), 0f64.into()),
            RowVector4::new(0f64.into(), 0f64.into(), 0f64.into(), 1f64.into()),
            RowVector4::new(0f64.into(), 0f64.into(), 1f64.into(), 0f64.into()),
        ]))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl ApproxEq for Qubit {
        type Epsilon = <f64 as ApproxEq>::Epsilon;

        fn default_epsilon() -> <Self as ApproxEq>::Epsilon {
            <f64 as ApproxEq>::default_epsilon()
        }

        fn default_max_relative() -> <Self as ApproxEq>::Epsilon {
            <f64 as ApproxEq>::default_max_relative()
        }

        fn default_max_ulps() -> u32 {
            <f64 as ApproxEq>::default_max_ulps()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: <Self as ApproxEq>::Epsilon,
            max_relative: <Self as ApproxEq>::Epsilon,
        ) -> bool {
            self.a.re.relative_eq(&other.a.re, epsilon, max_relative)
                && self.a.im.relative_eq(&other.a.im, epsilon, max_relative)
                && self.b.re.relative_eq(&other.b.re, epsilon, max_relative)
                && self.b.im.relative_eq(&other.b.im, epsilon, max_relative)
        }

        fn ulps_eq(
            &self,
            other: &Self,
            epsilon: <Self as ApproxEq>::Epsilon,
            max_ulps: u32,
        ) -> bool {
            self.a.re.ulps_eq(&other.a.re, epsilon, max_ulps)
                && self.a.im.ulps_eq(&other.a.im, epsilon, max_ulps)
                && self.b.re.ulps_eq(&other.b.re, epsilon, max_ulps)
                && self.b.im.ulps_eq(&other.b.im, epsilon, max_ulps)
        }
    }

    impl ApproxEq for QState {
        type Epsilon = <f64 as ApproxEq>::Epsilon;

        fn default_epsilon() -> <Self as ApproxEq>::Epsilon {
            <f64 as ApproxEq>::default_epsilon()
        }

        fn default_max_relative() -> <Self as ApproxEq>::Epsilon {
            <f64 as ApproxEq>::default_max_relative()
        }

        fn default_max_ulps() -> u32 {
            <f64 as ApproxEq>::default_max_ulps()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: <Self as ApproxEq>::Epsilon,
            max_relative: <Self as ApproxEq>::Epsilon,
        ) -> bool {
            self.state.iter().zip(other.state.iter()).all(|(c0, c1)| {
                c0.re.relative_eq(&c1.re, epsilon, max_relative)
                    && c0.im.relative_eq(&c1.im, epsilon, max_relative)
            })
        }

        fn ulps_eq(
            &self,
            other: &Self,
            epsilon: <Self as ApproxEq>::Epsilon,
            max_ulps: u32,
        ) -> bool {
            self.state.iter().zip(other.state.iter()).all(|(c0, c1)| {
                c0.re.ulps_eq(&c1.re, epsilon, max_ulps) && c0.im.ulps_eq(&c1.im, epsilon, max_ulps)
            })
        }
    }

    macro_rules! impl_qgate_relative_eq {
        ($T:ident) => {
            impl ApproxEq for $T {
                type Epsilon = <f64 as ApproxEq>::Epsilon;

                fn default_epsilon() -> <Self as ApproxEq>::Epsilon {
                    <f64 as ApproxEq>::default_epsilon()
                }

                fn default_max_relative() -> <Self as ApproxEq>::Epsilon {
                    <f64 as ApproxEq>::default_max_relative()
                }

                fn default_max_ulps() -> u32 {
                    <f64 as ApproxEq>::default_max_ulps()
                }

                fn relative_eq(
                    &self,
                    other: &Self,
                    epsilon: <Self as ApproxEq>::Epsilon,
                    max_relative: <Self as ApproxEq>::Epsilon,
                ) -> bool {
                    self.0.iter().zip(other.0.iter()).all(|(a, b)| {
                        a.re.relative_eq(&b.re, epsilon, max_relative)
                            && a.im.relative_eq(&b.im, epsilon, max_relative)
                    })
                }

                fn ulps_eq(
                    &self,
                    other: &Self,
                    epsilon: <Self as ApproxEq>::Epsilon,
                    max_ulps: u32,
                ) -> bool {
                    self.0.iter().zip(other.0.iter()).all(|(a, b)| {
                        a.re.ulps_eq(&b.re, epsilon, max_ulps)
                            && a.im.ulps_eq(&b.im, epsilon, max_ulps)
                    })
                }
            }
        };
    }

    impl_qgate_relative_eq!(QGate1);
    impl_qgate_relative_eq!(QGate2);

    fn mk_qstate(qs: &[bool]) -> QState {
        let v: Vec<_> = qs.iter()
            .map(|&b| if b { Qubit::ONE } else { Qubit::ZERO })
            .collect();
        QState::from_independent_qubits(&v)
    }

    #[test]
    fn test_qubit() {
        Qubit::new().validate();
        Qubit::ZERO.validate();
        Qubit::ONE.validate();
        Qubit::PLUS.validate();
        Qubit::MINUS.validate();

        assert_relative_eq!(
            Qubit::from_components(Qubit::ZERO.a, Qubit::ZERO.b),
            Qubit::ZERO
        );
        assert_relative_eq!(
            Qubit::from_components(Qubit::ONE.a, Qubit::ONE.b),
            Qubit::ONE
        );
        assert_relative_eq!(
            Qubit::from_components(Qubit::PLUS.a, Qubit::PLUS.b),
            Qubit::PLUS
        );
        assert_relative_eq!(
            Qubit::from_components(Qubit::MINUS.a, Qubit::MINUS.b),
            Qubit::MINUS
        );
    }

    struct ConstF64Rng(f64);
    impl Rng for ConstF64Rng {
        fn next_u32(&mut self) -> u32 {
            panic!()
        }
        fn next_u64(&mut self) -> u64 {
            panic!()
        }
        fn next_f32(&mut self) -> f32 {
            panic!()
        }
        fn next_f64(&mut self) -> f64 {
            self.0
        }
    }

    #[test]
    fn test_qubit_measure() {
        let mut rng = rand::thread_rng();

        fn measure(qubit: Qubit, n: usize, mut rng: &mut Rng) -> Vec<bool> {
            (0..n)
                .map(|_| QState::from_independent_qubits(&vec![qubit.clone()]).measure(&mut rng))
                .map(|v| {
                    assert_eq!(v.len(), 1);
                    v[0]
                })
                .collect()
        }

        fn count(bx: &[bool], b: bool) -> usize {
            bx.iter().filter(|&v| *v == b).count()
        }

        let n = 100;
        assert_eq!(count(&measure(Qubit::ZERO, n, &mut rng), false), n);
        assert_eq!(count(&measure(Qubit::ONE, n, &mut rng), true), n);
        assert_eq!(
            count(&measure(Qubit::PLUS, n, &mut ConstF64Rng(0.1)), false),
            n
        );
        assert_eq!(
            count(&measure(Qubit::PLUS, n, &mut ConstF64Rng(0.6)), true),
            n
        );
        assert_eq!(
            count(&measure(Qubit::MINUS, n, &mut ConstF64Rng(0.1)), false),
            n
        );
        assert_eq!(
            count(&measure(Qubit::MINUS, n, &mut ConstF64Rng(0.6)), true),
            n
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed: bits > 0")]
    fn test_n_to_bitvec_0_bits() {
        n_to_bitvec(0, 0);
    }

    #[test]
    fn test_n_to_bitvec() {
        assert_eq!(n_to_bitvec(1, 1), vec![true]);
        assert_eq!(n_to_bitvec(5, 4), vec![false, true, false, true]);
        assert_eq!(n_to_bitvec(22, 5), vec![true, false, true, true, false]);
    }

    #[test]
    fn test_qstate() {
        let mut rng = rand::thread_rng();

        assert_eq!(mk_qstate(&[false]).qubit_count(), 1);
        assert_eq!(mk_qstate(&[false, true]).qubit_count(), 2);

        assert_relative_eq!(
            mk_qstate(&[true]),
            QState {
                state: DVector::from_row_slice(
                    2,
                    &vec![
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 1f64, im: 0f64 },
                    ]
                ),
            }
        );
        assert_relative_eq!(
            mk_qstate(&[false, true]),
            QState {
                state: DVector::from_row_slice(
                    4,
                    &vec![
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 1f64, im: 0f64 },
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 0f64, im: 0f64 },
                    ]
                ),
            }
        );
        assert_relative_eq!(
            mk_qstate(&[false, true, true]),
            QState {
                state: DVector::from_row_slice(
                    8,
                    &vec![
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 1f64, im: 0f64 },
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 0f64, im: 0f64 },
                        Complex64 { re: 0f64, im: 0f64 },
                    ]
                ),
            }
        );

        fn check_measure_roundtrip(qubit_count: u32, mut rng: &mut Rng) {
            for i in 0..(1usize << qubit_count) {
                let qubits = n_to_bitvec(i, qubit_count);
                assert_eq!(mk_qstate(&qubits).measure(&mut rng), qubits);
            }
        }

        check_measure_roundtrip(1, &mut rng);
        check_measure_roundtrip(2, &mut rng);
        check_measure_roundtrip(3, &mut rng);
        check_measure_roundtrip(4, &mut rng);
    }

    #[test]
    fn test_hadamard() {
        let qs0 = QState::from_independent_qubits(&vec![Qubit::ZERO]);
        let qs1 = QState::from_independent_qubits(&vec![Qubit::ONE]);

        let qs0r = qs0.apply(&QGate1::hadamard());
        let qs1r = qs1.apply(&QGate1::hadamard());

        assert_relative_eq!(
            Qubit::PLUS,
            Qubit::from_components(qs0r.state[0], qs0r.state[1])
        );
        assert_relative_eq!(
            Qubit::MINUS,
            Qubit::from_components(qs1r.state[0], qs1r.state[1])
        );
    }

    #[test]
    fn test_swap() {
        let mut rng = rand::thread_rng();
        let qs = QState::from_independent_qubits(&vec![Qubit::ZERO, Qubit::ONE]);
        assert_eq!(
            qs.apply2(&QGate2::swap()).measure(&mut rng),
            vec![true, false]
        );
    }

    #[test]
    fn test_cnot() {
        fn check_cnot(qubits: Vec<Qubit>, expected: Vec<bool>, mut rng: &mut Rng) {
            let qs = QState::from_independent_qubits(&qubits);
            let result = qs.apply2(&QGate2::cnot()).measure(&mut rng);
            assert_eq!(result, expected);
        }
        let mut rng = rand::thread_rng();

        check_cnot(vec![Qubit::ZERO, Qubit::ZERO], vec![false, false], &mut rng);
        check_cnot(vec![Qubit::ZERO, Qubit::ONE], vec![false, true], &mut rng);
        check_cnot(vec![Qubit::ONE, Qubit::ZERO], vec![true, true], &mut rng);
        check_cnot(vec![Qubit::ONE, Qubit::ONE], vec![true, false], &mut rng);
    }

    #[test]
    fn test_pauli_z_eq_phase_shift_pi() {
        assert_relative_eq!(QGate1::pauli_z(), QGate1::phase_shift(std::f64::consts::PI));
    }
}
