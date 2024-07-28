use utils::random;

use ndarray::{Array1, Array, Array2};

use std::sync::Arc;
use std::fmt::Debug;

use mask::Mask;


pub struct DynamicsSet {
    pub f: Arc<dyn Fn(u32) -> Array1<f32>>
}

impl DynamicsSet  {
    pub fn vec(&self, n: usize) -> Vec<Array1<f32>> {
        let mut s = Vec::new();

        for i in 0..n as u32 {
            let v = (self.f)(i);
            s.push(v);
        }

        s
    }
    //pub fn matrix(&self, n: usize) -> Array1<Array1<f32>>

}

#[derive(Debug)]
pub struct ValueSet(pub Array2<f32>);

pub mod mask {
    use super::*;

    use std::ops::{Add, Sub};

    /// A mask consists of a function (i,j) -> {0,1}
    pub struct Mask {
        pub f: Arc<dyn Fn(u32, u32) -> bool>
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
}

pub mod op {
    use super::*;

    type LabelFn = Arc<dyn Fn(u32) -> u32>;

    /// The operator version of \rho, operating on the value set of connection probabilities
    pub fn p(v: ValueSet) -> Mask {
        Mask {
            f: Arc::new(
                    move |i, j| random::random_range((0.0, 1.0)) < v.0[[i as usize,j as usize]]
                )
        }
    }
    //fn sample(v: ValueSet<f32>) ->

    //fn disc(r: f32, dist: impl Space) -> Mask {
    //    Mask {
    //        f: Arc::new(
    //               move |i, j| dist(i, j) < r
    //           )
    //    }
    //}

    /// Expands each entry in c to a nxn block in a new matrix
    pub fn block(n: u32, m: Mask) -> Mask {
        let f1 = m.f.clone();

        Mask {
            f: Arc::new(
                   move |i, j| f1(i/n, j/n)
               )
        }
    }

    pub fn sbm(l: LabelFn, cpm: ValueSet) -> Mask {
        Mask {
            f: Arc::new(
                   move |i, j| random::random_range((0.0, 1.0)) < cpm.0[[l(i) as usize, l(j) as usize]]
               )
        }
    }

    pub fn label(dist: Vec<usize>, k: usize) -> LabelFn {
        let mut dist_map = vec![];
        for l in 0..k {
            dist_map.append(&mut vec![l; dist[l]]);
        }

        Arc::new(move |i| dist_map[i as usize] as u32)
    }
}

//trait Metric {
//    fn distance(i: u32, j: u32) -> f32;
//}
//
//type R2 = (f32, f32);
//
//struct R2Space;
//impl R2Space {
//    fn dist(g1: impl Fn(u32) -> R2, g2: impl Fn(u32) -> R2)-> impl Fn(u32, u32) -> f32 {
//        move |i,j| {
//            let (x1, y1) = g1(i);
//            let (x2, y2) = g2(j);
//
//            ((x1 - x2).powf(2.0) + (y1-y2).powf(2.0)).sqrt()
//        }
//    }
//}
//
//mod models {
//
//}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const N: usize = 10;

    #[test]
    fn mask_can_add() {
        let mask = mask::full() - mask::one_to_one();

        let matrix = mask.matrix(N);

        let id_m: Array2<u32> = Array::eye(N);
        let a: Array2<u32> = Array::ones((N, N)) - id_m;

        assert!(matrix == a);
    }

    #[test]
    fn mask_can_sub() {
        let mask = mask::empty() + mask::one_to_one();

        let matrix = mask.matrix(N);

        let id_m: Array2<u32> = Array::eye(N);

        assert!(matrix == id_m);
    }

    #[test]
    fn mask_p_op() {
        let v0 = ValueSet(Array::zeros((N, N)));
        let v1 = ValueSet(Array::ones((N, N)));

        let m0 = (op::p(v0)).matrix(N);
        let m1 = (op::p(v1)).matrix(N);

        let t0: Array2<u32> = Array::zeros((N, N));
        let t1: Array2<u32> = Array::ones((N, N));

        assert!(m0 == t0);
        assert!(m1 == t1);
    }

    //#[test]
    fn mask_block_op() {
        let mask = mask::one_to_one();

        let bm = op::block(2, m);
    }
}
