//! Representations are low-level containers for synaptic connections between neurons
//! that all other types of synapse are derived from.
//!

use crate::spikes::Spikes;
use crate::synapse::SynapticPotential;

use ndarray::{Array, Array1, Array2};

use std::collections::HashMap;
use std::convert::From;


/// A vector describing the neuron type of N neurons, where entry i is -1.0 if
/// neuron i is inhibitory, or 1.0 if it is excitatory.
type NeuronType = Array1<f32>;


pub enum Representation {
    Matrix(MatrixRepresentation),
    Map(MapRepresentation)
}

pub trait SynapseRepresentation {
    fn step(&mut self, input: &Spikes) -> SynapticPotential;
}

/// NOTE: Connections are stored as from->to;
/// that is, connections[i] contains weights of the neurons that neuron i projects TO.
pub struct MapRepresentation {
    connections: HashMap<usize, Vec<(usize, f32)>>,
    neuron_type: NeuronType
}

impl SynapseRepresentation for MapRepresentation {
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

impl MapRepresentation {
    pub fn new(connections: HashMap<usize, Vec<(usize, f32)>>, neuron_type: NeuronType) -> MapRepresentation {
        MapRepresentation {
            connections,
            neuron_type
        }
    }
}

impl From<&MatrixRepresentation> for MapRepresentation {
    fn from(item: &MatrixRepresentation) -> MapRepresentation {
        let mut connections: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

        for (from_neuron, weights) in item.weights.t().outer_iter().enumerate() {
            for (to_neuron, w) in weights.iter().enumerate() {
                connections.entry(from_neuron).or_insert(Vec::new()).push((to_neuron, *w));
            }
        }

        MapRepresentation::new(connections, item.neuron_type.to_owned())
    }
}


/// Connections betweenN neurons represented by an NXN matrix.
///
/// This means that entry W_jk is the weight from neuron k to neuron j
///
pub struct MatrixRepresentation {
    pub weights: Array2<f32>,
    pub neuron_type: NeuronType,
}

impl SynapseRepresentation for MatrixRepresentation {
    fn step(&mut self, input: &Spikes) -> SynapticPotential {
        let ns = &input.as_float() * &self.neuron_type;

        self.weights.dot(&ns)
    }
}

impl MatrixRepresentation {
    pub fn new(weight_matrix: Array2<f32>, neuron_type: NeuronType) -> MatrixRepresentation {
        assert!(weight_matrix.shape()[0] == neuron_type.shape()[0]);

        MatrixRepresentation {
            weights: weight_matrix,
            neuron_type
        }
    }

    /// Number of neurons
    pub fn neuron_count(&self) -> usize {
        self.weights.shape()[0]
    }

    /// Number of connections
    pub fn connection_count(&self) -> usize {
        self.weights.iter().fold(0usize, |acc, &x| if x != 0.0 {acc + 1} else {acc})
    }

    /// % of connections of all possible
    pub fn density(&self) -> f32 {
        self.connection_count() as f32 / self.neuron_count().pow(2) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    fn get_test_matrix_repr() -> MatrixRepresentation {
        let weights: Array2<f32> = array![[0.0, 2.0, 1.0],
                                          [1.0, 0.0, 0.0],
                                          [0.5, 0.8, 0.0]];

        let neuron_type: Array1<f32> = array![1.0, 1.0, -1.0];

        MatrixRepresentation::new(weights, neuron_type)
    }

    fn get_test_map_repr() -> MapRepresentation {
        let mut connections: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        let neuron_type: Array1<f32> = array![1.0, 1.0, -1.0];

        connections.insert(0, vec![]);
        connections.insert(1, vec![]);
        connections.insert(2, vec![]);

        connections.get_mut(&0).unwrap().push((1, 1.0));
        connections.get_mut(&0).unwrap().push((2, 0.5));
        connections.get_mut(&1).unwrap().push((0, 2.0));
        connections.get_mut(&1).unwrap().push((2, 0.8));
        connections.get_mut(&2).unwrap().push((0, 1.0));

        MapRepresentation::new(connections, neuron_type)
    }

    fn get_test_input() -> Spikes {
        let firing_state: Array1<bool> = array![true, false, true];

        Spikes { data: firing_state }
    }

    #[test]
    fn test_matrix_repr_output_correct() {
        let mut matrix_repr = get_test_matrix_repr();
        let mut i = get_test_input();

        let output = matrix_repr.step(&i);

        assert_eq!(output, array![-1.0, 1.0, 0.5]);
    }

    fn test_map_repr_output_correct() {
        let mut map_repr = get_test_map_repr();
        let mut i = get_test_input();

        let output = map_repr.step(&i);

        assert_eq!(output, array![-1.0, 1.0, 0.5]);
    }

    #[test]
    fn test_map_repr_from_matrix_repr() {
        let mut input = get_test_input();

        let mut matrix_repr = get_test_matrix_repr();

        let matrix_output = matrix_repr.step(&input);

        let mut map_repr = MapRepresentation::from(&matrix_repr);
        let map_output = map_repr.step(&input);

        assert_eq!(map_output, matrix_output);
    }
}
