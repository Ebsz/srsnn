use crate::model::FiringState;

use std::collections::HashMap;

use crate::utils::{random_matrix, random_range, random_sample};

use ndarray::{Array1, Array2, Array};

use ndarray_rand::rand_distr::StandardNormal;

/// Model of a set of synapses
pub trait Synapses { 
    fn step(&mut self, input: FiringState) -> Array1<f32>;
}

/// Synapses where connections are stored in a HashMap
/// NOTE: connections are stored as from->to, where connections[i] contains a Vec
/// of neurons that neuron i projects TO.
pub struct LinearSynapses {
    connections: HashMap<usize, Vec<(usize, f32)>>
}

impl LinearSynapses {
    /// Generate connections using the Erdős-Rényi model for random graphs,
    /// where each edge has an equal probability p of being included, independent of other edges
    pub fn from_probability(n: usize, p: f32) -> LinearSynapses {
        assert!(p > 0.0 && p < 1.0);

        let mut connections: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                if random_range((0.0, 1.0)) > (1.0 - p) {
                    let w = random_sample(StandardNormal);

                    connections.entry(i).or_insert(Vec::new()).push((j, w));
                }
            }
        }

        LinearSynapses {
            connections
        }
    }
}

impl Synapses for LinearSynapses {
    fn step(&mut self, input: FiringState) -> Array1<f32> {
        let mut output: Array1<f32> = Array::zeros(input.len());

        for neuron in input.firing() {
            // Indices of neurons that is projected to by the firing neuron
            let targets = self.connections.entry(neuron).or_insert(Vec::new());

            for (i, w) in targets {
                output[*i] += *w;
            }
        }

        output
    }
}

/// Synapses where connections between the N neurons are represented by an NxN matrix
/// where entry W_jk is the weight from neuron k to neuron j
pub struct MatrixSynapses {
    weights: Array2<f32>
}

impl Synapses for MatrixSynapses {
    fn step(&mut self, input: FiringState) -> Array1<f32> {
        self.weights.dot(&input.state)
    }
}

impl MatrixSynapses {
    pub fn new(n: usize) -> MatrixSynapses {
        let weights: Array2<f32> = random_matrix((n,n));

        //TODO: Assert I(W) = 0, ie. weights from a neuron to itself=0: 
        //      we don't want no self-feedback, yo.
        //      Can be done by W = (I(W) +1) % 2, or something.

        MatrixSynapses {
            weights
        }
    }
}
