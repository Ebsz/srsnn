use utils::random;

use ndarray::{Array, Array2};

use std::ops::{Add, Sub};
use std::sync::Arc;


pub type MaskFn = Arc<dyn Fn(u32, u32) -> bool>;

/// A mask consists of a function (i,j) -> {0,1}
pub struct Mask {
    pub f: MaskFn
}

impl Mask {
    // Output of this should be a u32 x [f32; n] matrix
    pub fn matrix(&self, n: usize) -> Array2<u32> {
        let mut m = Array::zeros((n,n));

        for (ix, v) in m.iter_mut().enumerate() {
            let i = (ix / n) as u32;
            let j = (ix % n) as u32;

            if (self.f)(i,j) {
                *v = 1;
            }
        }

        m
    }

    pub fn r_matrix(&self, n: usize, m: usize) -> Array2<u32> {
        let mut mx = Array::zeros((n,m));

        for (ix, v) in mx.iter_mut().enumerate() {
            let i = (ix / m) as u32;
            let j = (ix % m) as u32;

            if (self.f)(i,j) {
                *v = 1;
            }
        }

        mx
    }
}

impl Add for Mask {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let f1 = self.f.clone();
        let f2 = other.f.clone();

        Mask {
            f: Arc::new(move |i,j| f1(i,j) || f2(i,j))
        }
    }
}

impl Sub for Mask {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let f1 = self.f.clone();
        let f2 = other.f.clone();

        Mask {
            f: Arc::new(move |i,j| f1(i,j) && !f2(i,j))
        }
    }
}

pub fn empty() -> Mask {
    Mask { f: Arc::new(|_i, _j| false) }
}

pub fn full() -> Mask {
    Mask { f: Arc::new(|_i, _j| true) }
}

pub fn one_to_one() -> Mask {
    Mask { f: Arc::new(|i, j| i == j) }
}

pub fn random(p: f32) -> Mask {
    Mask {f: Arc::new(
        move |_i, _j| random::random_range((0.0, 1.0)) < p
    )}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    use utils::math;

    const N: usize = 10;

    #[test]
    fn mask_can_add() {
        let mask = full() - one_to_one();

        let matrix = mask.matrix(N);

        let id_m: Array2<u32> = Array::eye(N);
        let a: Array2<u32> = Array::ones((N, N)) - id_m;

        assert!(matrix == a);
    }

    #[test]
    fn mask_can_sub() {
        let mask = empty() + one_to_one();

        let matrix = mask.matrix(N);

        let id_m: Array2<u32> = Array::eye(N);

        assert!(matrix == id_m);
    }
}

