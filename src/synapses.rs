use crate::model::FiringState;

use ndarray::{Array1, Array2, Array};

use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::RandomExt;


/// Model of a set of synapses
pub trait Synapses { 
    fn step(&self, input: FiringState) -> Array1<f32>;
}

/// Synapse where connections between the N neurons are represented by an NxN matrix
/// where entry W_jk is the weight from neuron k to neuron j
pub struct MatrixSynapse {
    weights: Array2<f32>
}

impl Synapses for MatrixSynapse {
    fn step(&self, input: FiringState) -> Array1<f32> {
        self.weights.dot(&input.state)
    }
}

impl MatrixSynapse {
    pub fn new(n: usize) -> MatrixSynapse {

        let mut rng = StdRng::seed_from_u64(0);

        let mut weights: Array2<f32> = Array::random_using((n, n), StandardNormal, &mut rng);

        //TODO: Assert I(W) = 0, ie. weights from a neuron to itself=0: 
        //      we don't want no self-feedback, yo.
        //      Can be done by W = (I(W) +1) % 2, or something.

        MatrixSynapse {
            weights
        }
    }
}
