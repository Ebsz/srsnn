//! Functions for encoding real values as spikes

use crate::{random, math};

use ndarray::{Array, Array1, Array2, Zip};
use ndarray_rand::rand_distr::Uniform;


pub fn rate_encode(data: &Array1<f32>) -> Array1<f32> {
    let samples = random::random_vector(data.shape()[0], Uniform::new(0.0,1.0));

    data.iter().zip(samples).map(|(x, s)| if s < *x { 1.0 } else { 0.0 }).collect()
}

pub fn rate_encode_array(data: &Array2<f32>) -> Array2<f32> {
    let a = data.shape()[0];
    let b = data.shape()[1];

    let mut encoded: Array2<f32> = Array::zeros((a, b));

    let samples: Array2<f32> = random::random_matrix((a, b), Uniform::new(0.0,1.0));

    Zip::from(&mut encoded)
        .and(data)
        .and(&samples)
        .for_each(|e, d, s| *e = if s < d {1.0} else {0.0});

    encoded
}

pub fn time_encode(data: &Array1<f32>, max_t: usize) -> Array2<f32> {
    let max: f32 = math::maxf(data.as_slice().unwrap());
    let min: f32 = math::minf(data.as_slice().unwrap());

    let d = (data - min) / (max - min);

    //TODO: implement
    let out = Array::zeros((d.len(), max_t));

    // T * cos(d+i * pi/n)

    out
}
