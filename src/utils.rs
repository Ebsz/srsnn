use ndarray::{Array2, Array1, Array};

use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::RandomExt;

use ndarray_rand::rand_distr::{Distribution, StandardNormal};

use ndarray_rand::rand_distr::uniform::SampleUniform;
use std::cmp::PartialOrd;

use ndarray_rand::rand::Rng;

use std::ops::Range;

use std::cell::RefCell;


thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(09));

}

/// Generate a random matrix
// TODO: Rewrite with generic parameter to support matrices of any type
pub fn random_matrix(shape: (usize, usize) ) -> Array2<f32>{
    RNG.with(|rng| {
        Array::random_using(shape, StandardNormal, &mut (*rng.borrow_mut()))
    })
}

// TODO: change to use std::ops::Range for more natural syntax
/// Get a random sample from a specified range
pub fn random_range<T: SampleUniform + PartialOrd>(range: (T, T)) -> T {
    RNG.with(|rng| rng.borrow_mut().gen_range(range.0..range.1))
}


/// Sample from a distribution
pub fn random_sample<T, D: Distribution<T>> (dist: D) -> T {
    RNG.with(|rng| rng.borrow_mut().sample(dist))
}


// TODO: use this
//pub fn indices<T: Iterator>(x: T, pred: fn()) -> Vec<usize> {
//    x.iter().enumerate().filter(|(_, n)| **n != 0.0).map(|(i,_)| i).collect()
//}
