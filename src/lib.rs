extern crate alga;
#[macro_use]
extern crate approx;
extern crate nalgebra;
extern crate num_complex;
extern crate num_traits;
extern crate rand;

use std::fmt::{self, Debug, Display, Formatter};
use std::ops::{Add, Div, Mul, Sub};

use alga::general::{ClosedDiv, ClosedMul};
use approx::ApproxEq;
use nalgebra::*;
use num_complex::Complex64;
use num_traits::{One, Zero};
use rand::Rng;

#[derive(Clone)]
pub struct Qubit {
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
struct QStateExpr {
    state: DVector<Complex64>,
}

impl QStateExpr {
    fn from_qubits(qubits: &[Qubit]) -> QStateExpr {
        let expr = QStateExpr {
            state: qubits
                .iter()
                .map(|q| DVector::from_row_slice(&[q.a, q.b]))
                .fold(DVector::from_element(1, Complex64::one()), |acc, elem| {
                    acc.kronecker(&elem).into()
                }),
        };
        assert_eq!(qubit_count(&expr.state) as usize, qubits.len());
        expr
    }

    fn eval(self) -> QState {
        QState::from(self)
    }
}

impl<'a, 'b> Add<&'b QStateExpr> for &'a QStateExpr {
    type Output = QStateExpr;

    fn add(self, rhs: &'b QStateExpr) -> QStateExpr {
        QStateExpr {
            state: &self.state + &rhs.state,
        }
    }
}

impl<'a, T> Div<T> for &'a QStateExpr
where
    T: Scalar + ClosedDiv + Into<Complex64>,
{
    type Output = QStateExpr;

    fn div(self, rhs: T) -> QStateExpr {
        QStateExpr {
            state: &self.state / rhs.into(),
        }
    }
}

impl<'a, T> Mul<T> for &'a QStateExpr
where
    T: Scalar + ClosedMul + Into<Complex64>,
{
    type Output = QStateExpr;

    fn mul(self, rhs: T) -> QStateExpr {
        QStateExpr {
            state: &self.state * rhs.into(),
        }
    }
}

impl<'a, 'b> Sub<&'b QStateExpr> for &'a QStateExpr {
    type Output = QStateExpr;

    fn sub(self, rhs: &'b QStateExpr) -> QStateExpr {
        QStateExpr {
            state: &self.state - &rhs.state,
        }
    }
}

struct QState {
    state: DVector<Complex64>,
}

impl Debug for QState {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "QState({:?})", self.state)
    }
}

impl QState {
    fn from(expr: QStateExpr) -> QState {
        let qstate = QState { state: expr.state };
        qstate.validate();
        qstate
    }

    fn validate(&self) {
        assert_relative_eq!(
            self.state.iter().map(|c| c.norm_sqr()).sum(),
            1f64,
            max_relative = 0.000000000000001f64
        )
    }

    fn qubit_count(&self) -> u32 {
        qubit_count(&self.state)
    }

    fn apply(self, gate: &QGate) -> QState {
        assert!(
            gate.0.is_square(),
            "Matrix is not square: {:?}",
            gate.0.shape()
        );
        assert_eq!(gate.0.nrows(), self.state.len());
        QState {
            state: &gate.0 * self.state,
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

    #[allow(unused_variables)]
    fn measure_one(self, index: u32, rng: &mut Rng) -> (bool, QState) {
        unimplemented!()
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

fn qubit_count(state: &DVector<Complex64>) -> u32 {
    assert!(state.len().is_power_of_two());
    state.len().trailing_zeros()
}

struct QGate(DMatrix<Complex64>);

impl Debug for QGate {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "QGate({:?})", self.0)
    }
}

impl QGate {
    fn identity() -> QGate {
        let l = Complex64::one();
        let o = Complex64::zero();
        let m: DMatrix<Complex64> = DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[l, o]),
            RowDVector::from_row_slice(&[o, l]),
        ]);
        QGate(m)
    }

    fn hadamard() -> QGate {
        let m: DMatrix<Complex64> = DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[1f64.into(), 1f64.into()]),
            RowDVector::from_row_slice(&[1f64.into(), (-1f64).into()]),
        ]);
        QGate(m / Complex64::new(2f64.sqrt(), 0f64))
    }

    fn pauli_x() -> QGate {
        QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[0f64.into(), 1f64.into()]),
            RowDVector::from_row_slice(&[1f64.into(), 0f64.into()]),
        ]))
    }

    fn pauli_y() -> QGate {
        QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[0f64.into(), -Complex64::i()]),
            RowDVector::from_row_slice(&[Complex64::i(), 0f64.into()]),
        ]))
    }

    fn pauli_z() -> QGate {
        QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[1f64.into(), 0f64.into()]),
            RowDVector::from_row_slice(&[0f64.into(), (-1f64).into()]),
        ]))
    }

    fn phase_shift(phi: f64) -> QGate {
        QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[1f64.into(), 0f64.into()]),
            RowDVector::from_row_slice(&[0f64.into(), Complex64::from_polar(&1f64, &phi)]),
        ]))
    }

    fn swap() -> QGate {
        QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[1f64.into(), 0f64.into(), 0f64.into(), 0f64.into()]),
            RowDVector::from_row_slice(&[0f64.into(), 0f64.into(), 1f64.into(), 0f64.into()]),
            RowDVector::from_row_slice(&[0f64.into(), 1f64.into(), 0f64.into(), 0f64.into()]),
            RowDVector::from_row_slice(&[0f64.into(), 0f64.into(), 0f64.into(), 1f64.into()]),
        ]))
    }

    fn cnot() -> QGate {
        QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[1f64.into(), 0f64.into(), 0f64.into(), 0f64.into()]),
            RowDVector::from_row_slice(&[0f64.into(), 1f64.into(), 0f64.into(), 0f64.into()]),
            RowDVector::from_row_slice(&[0f64.into(), 0f64.into(), 0f64.into(), 1f64.into()]),
            RowDVector::from_row_slice(&[0f64.into(), 0f64.into(), 1f64.into(), 0f64.into()]),
        ]))
    }

    /// Compose quantum gates to act on more qubits in parallel
    fn par(&self, rhs: &QGate) -> QGate {
        QGate(self.0.kronecker(&rhs.0))
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

    impl_qgate_relative_eq!(QGate);

    fn mk_qstate(qs: &[bool]) -> QState {
        let v: Vec<_> = qs
            .iter()
            .map(|&b| if b { Qubit::ONE } else { Qubit::ZERO })
            .collect();
        QStateExpr::from_qubits(&v).eval()
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
                .map(|_| {
                    QStateExpr::from_qubits(&vec![qubit.clone()])
                        .eval()
                        .measure(&mut rng)
                }).map(|v| {
                    assert_eq!(v.len(), 1);
                    v[0]
                }).collect()
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
        let qs0 = QStateExpr::from_qubits(&vec![Qubit::ZERO]).eval();
        let qs1 = QStateExpr::from_qubits(&vec![Qubit::ONE]).eval();

        let qs0r = qs0.apply(&QGate::hadamard());
        let qs1r = qs1.apply(&QGate::hadamard());

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
        let qs = QStateExpr::from_qubits(&vec![Qubit::ZERO, Qubit::ONE]).eval();
        assert_eq!(
            qs.apply(&QGate::swap()).measure(&mut rng),
            vec![true, false]
        );
    }

    #[test]
    fn test_cnot() {
        fn check_cnot(qubits: Vec<Qubit>, expected: Vec<bool>, mut rng: &mut Rng) {
            let qs = QStateExpr::from_qubits(&qubits).eval();
            let result = qs.apply(&QGate::cnot()).measure(&mut rng);
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
        assert_relative_eq!(QGate::pauli_z(), QGate::phase_shift(std::f64::consts::PI));
    }

    #[test]
    fn test_qstateexpr() {
        let qse = &(&QStateExpr::from_qubits(&[Qubit::ZERO, Qubit::ONE])
            - &QStateExpr::from_qubits(&[Qubit::ONE, Qubit::ZERO]))
            * std::f64::consts::FRAC_1_SQRT_2;
        let expected = DVector::from_row_slice(
            &vec![
                0f64,
                std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2,
                0f64,
            ].iter()
            .map(|x| x.into())
            .collect() as &Vec<Complex64>,
        );
        assert_eq!(qse.state, expected)
    }

    #[test]
    fn test_gate_identity_identity() {
        let identity = QGate::identity();
        let identity_sqr = identity.par(&identity);
        let l = Complex64::one();
        let o = Complex64::zero();
        let expected = QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[l, o, o, o]),
            RowDVector::from_row_slice(&[o, l, o, o]),
            RowDVector::from_row_slice(&[o, o, l, o]),
            RowDVector::from_row_slice(&[o, o, o, l]),
        ]));
        assert_relative_eq!(identity_sqr, expected);

        let qs = QStateExpr::from_qubits(&vec![Qubit::ZERO, Qubit::ONE]).eval();
        let qsr = qs.apply(&identity_sqr);
        assert_relative_eq!(
            qsr,
            QStateExpr::from_qubits(&vec![Qubit::ZERO, Qubit::ONE]).eval()
        );
    }

    #[test]
    fn test_gate_hadamard_identity() {
        let hadamard = QGate::hadamard();
        let identity = QGate::identity();
        let hadamard_identity = hadamard.par(&identity);
        let v = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0f64);
        let o = Complex64::zero();
        let expected = QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[v, o, v, o]),
            RowDVector::from_row_slice(&[o, v, o, v]),
            RowDVector::from_row_slice(&[v, o, -v, o]),
            RowDVector::from_row_slice(&[o, v, o, -v]),
        ]));
        assert_relative_eq!(hadamard_identity, expected);

        let qs = QStateExpr::from_qubits(&vec![Qubit::ZERO, Qubit::ONE]).eval();
        let qsr = qs.apply(&hadamard_identity);
        assert_relative_eq!(
            qsr,
            QStateExpr::from_qubits(&vec![Qubit::PLUS, Qubit::ONE]).eval()
        );
    }

    #[test]
    fn test_gate_identity_hadamard() {
        let hadamard = QGate::hadamard();
        let identity = QGate::identity();
        let identity_hadamard = identity.par(&hadamard);
        let v = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0f64);
        let o = Complex64::zero();
        let expected = QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[v, v, o, o]),
            RowDVector::from_row_slice(&[v, -v, o, o]),
            RowDVector::from_row_slice(&[o, o, v, v]),
            RowDVector::from_row_slice(&[o, o, v, -v]),
        ]));
        assert_relative_eq!(identity_hadamard, expected);

        let qs = QStateExpr::from_qubits(&vec![Qubit::ZERO, Qubit::ONE]).eval();
        let qsr = qs.apply(&identity_hadamard);
        assert_relative_eq!(
            qsr,
            QStateExpr::from_qubits(&vec![Qubit::ZERO, Qubit::MINUS]).eval()
        );
    }

    #[test]
    fn test_gate_hadamard_hadamard() {
        let hadamard = QGate::hadamard();
        let hadamard_sqr = hadamard.par(&hadamard);
        let v = Complex64::new(0.5f64, 0f64);
        let expected = QGate(DMatrix::from_rows(&[
            RowDVector::from_row_slice(&[v, v, v, v]),
            RowDVector::from_row_slice(&[v, -v, v, -v]),
            RowDVector::from_row_slice(&[v, v, -v, -v]),
            RowDVector::from_row_slice(&[v, -v, -v, v]),
        ]));
        assert_relative_eq!(hadamard_sqr, expected);

        let qs = QStateExpr::from_qubits(&vec![Qubit::ZERO, Qubit::ONE]).eval();
        let qsr = qs.apply(&hadamard_sqr);
        assert_relative_eq!(
            qsr,
            QStateExpr::from_qubits(&vec![Qubit::PLUS, Qubit::MINUS]).eval()
        );
    }
}
