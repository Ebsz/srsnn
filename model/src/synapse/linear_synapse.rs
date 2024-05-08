//! Synapses where connections are stored in a HashMap
//!

use crate::synapse::{Synapse, SynapticPotential};
use crate::spikes::Spikes;
use ndarray::{Array1, Array};


use std::collections::HashMap;


/// NOTE: connections are stored as from->to, that is,
/// connections[i] contains weights of the neurons
/// that neuron i projects _TO_.
pub struct LinearSynapse {
    connections: HashMap<usize, Vec<(usize, f32)>>
}

impl Synapse for LinearSynapse {
    fn step(&mut self, input: &Spikes) -> SynapticPotential {
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
    pub fn new(connections: HashMap<usize, Vec<(usize, f32)>>) -> LinearSynapse{
        LinearSynapse {
            connections
        }
    }
}
