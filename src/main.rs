#[macro_use]
extern crate approx;
extern crate nalgebra;
extern crate num_complex;
extern crate num_traits;
extern crate rand;

use std::fmt::{self, Display, Formatter};

use nalgebra::*;
use num_complex::{Complex, Complex64};
use num_traits::identities::One;
use num_traits::identities::Zero;
use rand::Rng;

fn main() {
    println!("{}", QGate1::hadamard().apply(&QBit::zero()));
}

// TODO This should be a Unit<Vector2<Complex<f64>>>
#[derive(Debug)]
struct QBit {
    a: Complex64,
    b: Complex64,
}

impl QBit {
    fn new() -> QBit {
        QBit::zero()
    }

    /// |0>
    fn zero() -> QBit {
        QBit::from_components(Complex64::new(1f64, 0f64), Complex64::new(0f64, 0f64))
    }

    /// |1>
    fn one() -> QBit {
        QBit::from_components(Complex64::new(0f64, 0f64), Complex64::new(1f64, 0f64))
    }

    fn from_components(a: Complex64, b: Complex64) -> QBit {
        let q = QBit { a, b };
        q.validate();
        q
    }

    fn validate(&self) {
        self.prob();
    }

    fn prob(&self) -> (f64, f64) {
        let (zp, op) = (self.a.norm_sqr(), self.b.norm_sqr());
        assert_relative_eq!(zp + op, 1f64);
        (zp, op)
    }

    fn measure<R: Rng>(self, rng: &mut R) -> QBit {
        self.validate();
        let (zp, op) = self.prob();
        let r = rng.gen::<f64>();
        if r < zp {
            QBit::zero()
        } else {
            QBit::one()
        }
    }
}

fn format_complex(c: &Complex64) -> String {
    return if relative_eq!(c.im, 0f64) {
        format!("{}", c.re)
    } else {
        format!("({}+{}i)", c.re, c.im)
    };
}

impl Display for QBit {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        let (zp, op) = self.prob();
        if relative_eq!(zp, 1f64) {
            write!(f, "|0>");
        } else if relative_eq!(op, 1f64) {
            write!(f, "|1>");
        } else {
            let a = format_complex(&self.a);
            let b = format_complex(&self.b);
            write!(f, "{}|0> + {}|1>", a, b);
        }
        Ok(())
    }
}

struct QGate1(Matrix2<Complex64>);

impl QGate1 {
    fn hadamard() -> QGate1 {
        let m: Matrix2<Complex64> = Matrix2::from_rows(&[
            RowVector2::new(1f64.into(),   1f64.into()),
            RowVector2::new(1f64.into(), (-1f64).into()),
        ]);
        QGate1(m / Complex64::from(2f64.sqrt()))
    }

    fn apply(&self, q: &QBit) -> QBit {
        let v = self.0 * Vector2::new(q.a, q.b);
        let qr = QBit::from_components(v[0], v[1]);
        qr
    }
}

struct QGate2(Matrix4<Complex64>);

impl QGate2 {
    fn cnot() -> QGate2 {
        QGate2(Matrix4::from_rows(&[
            RowVector4::new(1f64.into(), 0f64.into(), 0f64.into(), 0f64.into()),
            RowVector4::new(0f64.into(), 1f64.into(), 0f64.into(), 0f64.into()),
            RowVector4::new(0f64.into(), 0f64.into(), 0f64.into(), 1f64.into()),
            RowVector4::new(0f64.into(), 0f64.into(), 1f64.into(), 0f64.into()),
        ]))
    }

    fn apply(&self, q1: &QBit, q2: &QBit) -> (QBit, QBit) {
        let v = self.0 * Vector4::new(q1.a, q1.b, q2.a, q2.b);
        let qr1 = QBit::from_components(v[0], v[1]);
        let qr2 = QBit::from_components(v[2], v[3]);
        (qr1, qr2)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let mut rng = rand::thread_rng();
        let q = QBit::zero();

        let qq = q.measure(&mut rng);
        assert!(false, "{:?}", qq);
    }

    #[test]
    fn test_cnot() {
        let mut rng = rand::thread_rng();
        let q0 = QBit::zero();
        let q1 = QBit::one();

        let (qq0, qq1) = QGate2::cnot().apply(&q0, &q1);
        assert!(false, "CNot({:?}, {:?}) = ({:?}, {:?})", q0, q1, qq0, qq1);
    }
}
