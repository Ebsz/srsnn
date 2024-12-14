pub mod mask;
pub mod op;

use ndarray::{s, array, Array, Array1, Array2};

use std::sync::Arc;

use mask::Mask;


pub struct NetworkSet {
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

    pub fn r_matrix(&self, n: usize, m: usize) -> Array2<f32> {
        let mut mx = Array::zeros((n,m));

        for (ix, v) in mx.iter_mut().enumerate() {
            let i = (ix / m) as u32;
            let j = (ix % m) as u32;

            *v = (self.f)(i,j);
        }

        mx
    }

    ///// Merge a value set of values R^a with a value set of values R^b into a value set
    ///// of values R^(a+b), where the first values
    ///// TODO: This can instead be accomplished by implementing Add for ValueSet
    //pub fn merge(v1: ValueSet, v2: ValueSet) -> ValueSet {
    //    let f1 = v1.f.clone();
    //    let f2 = v2.f.clone();

    //    ValueSet {
    //        f: Arc::new(move |i,j| f1(i,j)  f2(i,j))
    //    }
    //}
}

pub type NeuronFn = Arc<dyn Fn(u32) -> Array1<f32>>;

/// Neuron sets are functions on single neuron indices, and capture properties of individual neurons
pub struct NeuronSet {
    pub f: NeuronFn
}

impl NeuronSet  {
    pub fn from_value(v: Array2<f32>) -> Self {
        NeuronSet {
            f: Arc::new( move |i| v.slice(s![i as usize, ..]).to_owned() )
        }
    }

    pub fn vec(&self, n: usize) -> Vec<Array1<f32>> {
        let mut s = Vec::new();

        for i in 0..n as u32 {
            let v = (self.f)(i);
            s.push(v);
        }

        s
    }
}

pub type NeuronMaskFn = Arc<dyn Fn(u32) -> bool>;

#[derive(Clone)]
pub struct NeuronMask {
    pub f: NeuronMaskFn
}

impl Into<NeuronSet> for NeuronMask {
    fn into(self) -> NeuronSet {
        let f = self.f;
        NeuronSet {
            f: Arc::new(move |i| array![(if f(i) { 1.0 } else  { 0.0 } )])
        }
    }
}
