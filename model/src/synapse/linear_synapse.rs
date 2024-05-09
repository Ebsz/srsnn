//! Synapses where connections are stored in a HashMap
//!

use crate::spikes::Spikes;
use crate::synapse::{Synapse, SynapticPotential};
use crate::synapse::matrix_synapse::MatrixSynapse;

use ndarray::{Array1, Array};

use std::collections::HashMap;
use std::convert::From;


/// NOTE: connections are stored as from->to, that is,
/// connections[i] contains weights of the neurons
/// that neuron i projects _TO_.
pub struct LinearSynapse {
    connections: HashMap<usize, Vec<(usize, f32)>>,
    neuron_type: Array1<f32>
}

impl Synapse for LinearSynapse {
    fn step(&mut self, input: &Spikes) -> SynapticPotential {
        let mut output: Array1<f32> = Array::zeros(input.len());

        for neuron in input.firing() {
            // Indices of neurons that is projected to by the firing neuron
            let targets = self.connections.entry(neuron).or_insert(Vec::new());

            for (i, w) in targets {
                output[*i] += *w * self.neuron_type[neuron];
            }
        }

        output
    }
}

impl LinearSynapse {
    pub fn new(connections: HashMap<usize, Vec<(usize, f32)>>, neuron_type: Array1<f32>) -> LinearSynapse {
        LinearSynapse {
            connections,
            neuron_type
        }
    }
}

impl From<MatrixSynapse> for LinearSynapse {
    fn from(item: MatrixSynapse) -> LinearSynapse {
        let mut connections: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

        for (from_neuron, weights) in item.weights.t().outer_iter().enumerate() {
            for (to_neuron, w) in weights.iter().enumerate() {
                connections.entry(from_neuron).or_insert(Vec::new()).push((to_neuron, *w));
            }
        }

        LinearSynapse::new(connections, item.neuron_type)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{array, Array2};

    #[test]
    fn test_linear_synapse() {
        let n = 3;
        let mut connections: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

        connections.insert(0, vec![]);
        connections.insert(1, vec![]);
        connections.insert(2, vec![]);

        connections.get_mut(&0).unwrap().push((1, 1.0));
        connections.get_mut(&0).unwrap().push((2, 0.5));
        connections.get_mut(&1).unwrap().push((0, 2.0));
        connections.get_mut(&1).unwrap().push((2, 0.8));
        connections.get_mut(&2).unwrap().push((0, 1.0));

        let neuron_type: Array1<f32> = array![1.0, 1.0, -1.0];

        let firing_state: Array1<bool> = array![true, false, true];

        let input = Spikes {
            data: firing_state
        };

        let mut s = LinearSynapse::new(connections, neuron_type);

        let output = s.step(&input);

        assert_eq!(output, array![-1.0, 1.0, 0.5]);
    }

    #[test]
    fn test_from_matrix_synapse() {
        let weights: Array2<f32> = array![[0.0, 2.0, 1.0],
                                          [1.0, 0.0, 0.0],
                                          [0.5, 0.8, 0.0]];

        let neuron_type: Array1<f32> = array![1.0, 1.0, -1.0];

        let firing_state: Array1<bool> = array![true, false, true];

        let input = Spikes {
            data: firing_state
        };

        let mut matrix_synapse = MatrixSynapse::new(weights, neuron_type);
        let matrix_output = matrix_synapse.step(&input);

        let mut linear_synapse = LinearSynapse::from(matrix_synapse);
        let linear_output = linear_synapse.step(&input);

        assert_eq!(linear_output, matrix_output);
    }
}
