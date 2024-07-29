use utils::random;

use ndarray::{Array1, Array, Array2};

use std::sync::Arc;
use std::fmt::Debug;

use mask::Mask;

use object::{Object, CSAObject};


pub mod object {
    use super::*;

    pub enum Object<D,R,O: Clone> {
        Function(Arc<dyn Fn(D) -> R>),
        Value(O)
    }

    pub trait CSAObject<D, R, O: Clone> {
        fn from_fn(f: Arc<dyn Fn(D) -> R>)  -> Self;
        fn from_value(v: O) -> Self;

        fn get(&self, n: usize) -> O {
            match self.obj() {
                Object::Value(o) => { o.clone() },
                Object::Function(f) => { Self::generate(f, n) }
            }
        }

        fn obj(&self) -> &Object<D,R,O>;
        fn generate(f: &Arc<dyn Fn(D) -> R>, n: usize) -> O;
    }

    //pub type MaskObject = Object<(u32, u32), bool, Array2<u32>>;
    //pub type ValueObject = Object<(u32, u32), f32, Array2<f32>>;

    //pub struct ValueObject {
    //    o: Object<(u32, u32), f32, Array2<f32>>
    //}

    //impl CSAObject<(u32, u32), f32, Array2<f32>> for ValueObject {
    //    fn from_value(v: Array2<f32>) -> Self {
    //        ValueSet {
    //            o: Object::Value(v)
    //        }
    //    }
    //
    //    fn from_fn(f: Arc<dyn Fn((u32, u32)) -> f32>) -> Self {
    //        ValueSet {
    //            o: Object::Function(f)
    //        }
    //    }

    //    fn obj(&self) -> &Object<(u32, u32), f32, Array2<f32>> {
    //        &self.o
    //    }

    //    fn generate(f: &Arc<dyn Fn((u32, u32)) -> f32>, n: usize) -> Array2<f32> {
    //        let mut m: Array2<f32> = Array::zeros((n, n));

    //        for (ix, v) in m.iter_mut().enumerate() {
    //            let i = (ix / n) as u32;
    //            let j = (ix % n) as u32;

    //            *v = (f)((i, j));
    //        }

    //        m
    //    }
    //}


    //pub struct Mask {
    //    o: MaskObject
    //}

    //impl CSAObject<(u32, u32), bool, Array2<u32>> for Mask
    //{
    //    fn obj(&self) -> &MaskObject {
    //        &self.o
    //    }
    //    fn from_fn(f: &Arc<dyn Fn((u32, u32)) -> bool>, n: usize) -> Array2<u32> {
    //        let mut m = Array::zeros((n,n));

    //        for (ix, v) in m.iter_mut().enumerate() {
    //            let i = (ix / n) as u32;
    //            let j = (ix % n) as u32;

    //            if (f)((i,j)) {
    //                *v = 1;
    //            }
    //        }

    //        m
    //    }
    //}
    //pub type DynamicsObject = Object<u32, Array1<f32>, Vec<Array1<f32>>>;
    //
    //pub struct DynamicsSet {
    //    o: DynamicsObject
    //}
    //
    //impl CSAObject<u32, Array1<f32>, Vec<Array1<f32>>> for DynamicsSet {
    //    fn obj(&self) -> &DynamicsObject {
    //        &self.o
    //    }
    //
    //    fn from_fn(f: &Arc<dyn Fn(u32) -> Array1<f32>>, n: usize) -> Vec<Array1<f32>> {
    //        let mut s = Vec::new();
    //
    //        for i in 0..n as u32 {
    //            let v = f(i);
    //            s.push(v);
    //        }
    //
    //        s
    //    }
    //}
}

type ValueFn = Arc<dyn Fn(u32, u32) -> f32>;

pub struct ConnectionSet {
    pub m: Mask,
    pub v: Vec<ValueSet>
}

pub struct ValueSet {
    pub f: ValueFn
}

impl ValueSet {
    pub fn from_value(v: Array2<f32>) -> Self {
        ValueSet {
            f: Arc::new(move |i,j| v[[ i as usize, j as usize ]])
        }
    }

    pub fn matrix(&self, n: usize) -> Array2<f32> {
        let mut m = Array::zeros((n,n));

        for (ix, v) in m.iter_mut().enumerate() {
            let i = (ix / n) as u32;
            let j = (ix % n) as u32;

            *v = (self.f)(i,j);
        }

        m
    }
}

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
        let f = v.f;
        Mask {
            f: Arc::new(
                    move |i, j| random::random_range((0.0, 1.0)) < f(i, j)
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
        let f = cpm.f;

        Mask {
            f: Arc::new(
                   move |i, j| random::random_range((0.0, 1.0)) < f(l(i), l(j))
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
    use ndarray::{array, s};

    use utils::math;

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
        let v0 = ValueSet::from_value(Array::zeros((N, N)));
        let v1 = ValueSet::from_value(Array::ones((N, N)));

        let m0 = (op::p(v0)).matrix(N);
        let m1 = (op::p(v1)).matrix(N);

        let t0: Array2<u32> = Array::zeros((N, N));
        let t1: Array2<u32> = Array::ones((N, N));

        assert!(m0 == t0);
        assert!(m1 == t1);
    }

    #[test]
    fn sbm_op() {
        let n = 10;
        let k = 2;
        let p = vec![0.5, 0.5];

        let dist = math::distribute(n, &p);
        println!("{:?}", dist);

        let labels = op::label(dist, k);

        let cpm = ValueSet::from_value(array![[0.0,1.0,],
                                              [1.0,0.0,]]);

        let m = op::sbm(labels, cpm);

        let mx = m.matrix(10);

        let a = mx.slice(s![..4,..4]); // top left
        let b = mx.slice(s![..4,(-4)..]); // top right
        let c = mx.slice(s![(-4)..,..4]); // bottom left
        let d = mx.slice(s![(-4)..,(-4)..]); // bottom right

        assert!(a.iter().all(|x| *x == 0));
        assert!(b.iter().all(|x| *x == 1));
        assert!(c.iter().all(|x| *x == 1));
        assert!(d.iter().all(|x| *x == 0));
    }
}
