use std::cmp::PartialOrd;
use std::cell::RefCell;

use ndarray::{Array1, Array2, Array};

use ndarray_rand::RandomExt;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::rand_distr::uniform::SampleUniform;

use rand::seq::SliceRandom;


pub const SEED: u64 = 1337;

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(SEED));
}

pub fn random_vector<D: Distribution<f32>>(size: usize, dist: D) -> Array1<f32> {
    RNG.with(|rng| {
        Array::random_using(size, dist, &mut (*rng.borrow_mut()))
    })
}

/// Generate a random matrix
pub fn random_matrix<T, D: Distribution<T>>(shape: (usize, usize), dist: D) -> Array2<T>{
    RNG.with(|rng| {
        Array::random_using(shape, dist, &mut (*rng.borrow_mut()))
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

/// Choose a random element from a Vec
pub fn random_choice<T> (v: &Vec<T>) -> &T {
    assert!(v.len() != 0);

    RNG.with(|rng| v.choose(&mut (*rng.borrow_mut()))).unwrap()
}

// TODO: use this
//pub fn indices<T: Iterator>(x: T, pred: fn()) -> Vec<usize> {
//    x.iter().enumerate().filter(|(_, n)| **n != 0.0).map(|(i,_)| i).collect()
//}
