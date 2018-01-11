#[macro_use]
extern crate approx;
extern crate nalgebra;
extern crate num_complex;
extern crate num_traits;
extern crate rand;

use nalgebra::*;
use num_complex::{Complex, Complex64};
use num_traits::identities::One;
use num_traits::identities::Zero;
use rand::Rng;

fn main() {
    println!("Hello, world! {:?}", QBit::one());
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
        let q = QBit {
            a: Complex64::new(1f64, 0f64),
            b: Complex64::new(0f64, 0f64),
        };
        q.validate();
        q
    }

    /// |1>
    fn one() -> QBit {
        let q = QBit {
            a: Complex64::new(0f64, 0f64),
            b: Complex64::new(1f64, 0f64),
        };
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

struct CNot(Matrix4<Complex64>);

impl CNot {
    fn new() -> CNot {
        CNot(Matrix4::new(
            1f64.into(),
            0f64.into(),
            0f64.into(),
            0f64.into(),
            0f64.into(),
            1f64.into(),
            0f64.into(),
            0f64.into(),
            0f64.into(),
            0f64.into(),
            0f64.into(),
            1f64.into(),
            0f64.into(),
            0f64.into(),
            1f64.into(),
            0f64.into(),
        ))
    }

    fn apply(&self, q1: &QBit, q2: &QBit) -> (QBit, QBit) {
        let v = self.0 * Vector4::new(q1.a, q1.b, q2.a, q2.b);
        (QBit { a: v[0], b: v[1] }, QBit { a: v[2], b: v[3] })
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
}
