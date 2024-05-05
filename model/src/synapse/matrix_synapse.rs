use crate::synapse::Synapse;
use crate::spikes::Spikes;

use ndarray::{Array1, Array2};

use std::fmt;

/// A synapse where connections where connections between the N neurons
/// are represented by an NxN matrix; entry W_jk is the weight from neuron k to neuron j
///
/// neuron_type is a vector of length N, where entry i is -1 if
/// neuron i is inhibitory, or 1 if it is excitatory.
pub struct MatrixSynapse {
    weights: Array2<f32>,
    neuron_type: Array1<f32>,
}

impl Synapse for MatrixSynapse {
    fn step(&mut self, input: &Spikes) -> Array1<f32> {
        let ns = &input.as_float() * &self.neuron_type;

        self.weights.dot(&ns)
    }
}

impl MatrixSynapse {
    pub fn new(weight_matrix: Array2<f32>, neuron_type: Array1<f32>) -> MatrixSynapse {
        assert!(weight_matrix.shape()[0] == neuron_type.shape()[0]);

        MatrixSynapse {
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

impl fmt::Display for MatrixSynapse {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MatrixSynapse<N: {}, c :{}, d: {}>", self.neuron_count(), self.connection_count(), self.density())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synapse::Synapse;
    use crate::spikes::Spikes;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_matrix_synapse_output_correct() {
        let weights: Array2<f32> = array![[0.0, 2.0, 1.0],
                                          [1.0, 0.0, 0.0],
                                          [0.5, 0.8, 0.0]];

        let neuron_type: Array1<f32> = array![1.0, 1.0, -1.0];
        let firing_state: Array1<bool> = array![true, false, true];

        let input = Spikes {
            data: firing_state
        };

        let mut s = MatrixSynapse::new(weights, neuron_type);

        let output = s.step(&input);

        assert_eq!(output, array![-1.0, 1.0, 0.5]);
    }
}
