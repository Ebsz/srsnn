//! A synapse where connections where connections between the N neurons
//! are represented by an NxN matrix; entry W_jk is the weight from neuron k to neuron j
///
/// neuron_type is a vector of length N, where entry i is -1 if
/// neuron i is inhibitory, or 1 if it is excitatory.
///

use crate::model::synapse::Synapse;
use crate::model::spikes::Spikes;
use crate::utils::{random_matrix};

use ndarray::{Array1, Array2, Array};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};

pub struct MatrixSynapse {
    weights: Array2<f32>,
    neuron_type: Array1<f32>
}

impl Synapse for MatrixSynapse {
    fn step(&mut self, input: &Spikes) -> Array1<f32> {
        let ns = &input.data * &self.neuron_type;

        self.weights.dot(&ns)
    }
}

impl MatrixSynapse {
    pub fn from_matrix(weight_matrix: Array2<f32>, neuron_type: Array1<f32>) -> MatrixSynapse {
        assert!(weight_matrix.shape()[0] == neuron_type.shape()[0]);

        MatrixSynapse {
            weights: weight_matrix,
            neuron_type
        }
    }

    /// Creates a set of randomly connected synapses, with a fraction set
    /// to be inhibitory.
    pub fn from_probability(n: usize, p: f32, inhibitory: Array1<bool>) -> MatrixSynapse{
        let mut weights: Array2<f32> = random_matrix((n,n), StandardNormal)
            .mapv(|x| x.abs());

        let mut enabled_connections: Array2<f32> = random_matrix((n,n), Uniform::new(0.0, 1.0))
            .mapv(|x| if x > (1.0 - p) {1.0} else {0.0});

        // Remove connections from a neuron to itself, ie. diagonal entries.
        let e: Array2<f32> = (Array::eye(n) + 1.0) % 2.0;
        enabled_connections = &enabled_connections * &e;

        weights = weights * &enabled_connections;

        let neuron_type: Array1<f32> = inhibitory.mapv(|i| if i {-1.0} else {1.0});

        Self::from_matrix(weights, neuron_type)
    }

    //pub fn random_matrix(n: usize) -> MatrixSynapse {
    //    let weights: Array2<f32> = random_matrix((n,n), StandardNormal);
    //    weights.mapv(|x| x.abs());

    //    // TODO: TMP: Find out how this should be initialized
    //    let neuron_type: Array1<f32> = Array::ones(n);

    //    MatrixSynapse {
    //        weights,
    //        neuron_type
    //    }
    //}

    pub fn neuron_count(&self) -> usize {
        self.weights.shape()[0]
    }
}

#[cfg(test)]
mod tests {
    use crate::model::synapse::matrix_synapse::MatrixSynapse;
    use crate::model::synapse::Synapse;
    use crate::model::spikes::Spikes;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_matrix_synapse_output_correct() {
        let weights: Array2<f32> = array![[0.0, 2.0, 1.0],
                                          [1.0, 0.0, 0.0],
                                          [0.5, 0.8, 0.0]];

        let neuron_type: Array1<f32> = array![1.0, 1.0, -1.0];
        let firing_state: Array1<f32> = array![1.0, 0.0, 1.0];

        let input = Spikes {
            data: firing_state
        };

        let mut s = MatrixSynapse::from_matrix(weights, neuron_type);

        let output = s.step(&input);

        assert_eq!(output, array![-1.0, 1.0, 0.5]);
    }
}
