//! Synapses where connections are stored in a HashMap
//!

use crate::model::synapse::Synapse;
use crate::model::spikes::Spikes;
use crate::utils::{random_range, random_sample};
use ndarray::{Array1, Array};

use std::collections::HashMap;

use ndarray_rand::rand_distr::StandardNormal;


/// NOTE: connections are stored as from->to, that is,
/// connections[i] contains weights of the neurons
/// that neuron i projects _TO_.
pub struct LinearSynapse {
    connections: HashMap<usize, Vec<(usize, f32)>>
}

impl Synapse for LinearSynapse {
    fn step(&mut self, input: &Spikes) -> Array1<f32> {
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

impl LinearSynapse {
    /// Generate connections using the Erdős-Rényi model for random graphs,
    /// where each edge has an equal probability p of being included, independent of other edges
    pub fn from_probability(n: usize, p: f32) -> LinearSynapse {
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

        LinearSynapse {
            connections
        }
    }
}
