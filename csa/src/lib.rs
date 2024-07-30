pub mod mask;
pub mod op;

use ndarray::{Array1, Array, Array2};

use std::sync::Arc;

use mask::Mask;

//pub struct NeuralSet {
//    pub m: Mask,
//    pub m_in: Mask,
//    pub v: Vec<ValueSet>,
//    pub d: NeuronSet,
//    pub ds: Vec<NeuronSet>,
//}

pub struct NeuralSet {
    pub m: Mask,
    pub v: Vec<ValueSet>,
    pub d: Vec<NeuronSet>
}

pub struct ConnectionSet {
    pub m: Mask,
    pub v: Vec<ValueSet>,
}

pub type ValueFn = Arc<dyn Fn(u32, u32) -> f32>;

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

pub type NeuronFn = Arc<dyn Fn(u32) -> Array1<f32>>;

pub struct NeuronSet {
    pub f: NeuronFn
}

impl NeuronSet  {
    pub fn vec(&self, n: usize) -> Vec<Array1<f32>> {
        let mut s = Vec::new();

        for i in 0..n as u32 {
            let v = (self.f)(i);
            s.push(v);
        }

        s
    }
}


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
