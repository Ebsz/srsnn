//! gen.rs
//!
//! Functions for randomly generating models


pub mod synapse_gen {
    use crate::model::synapse::matrix_synapse::MatrixSynapse;
    use crate::utils::random_matrix;

    use ndarray_rand::rand_distr::{StandardNormal, Uniform};
    use ndarray::{Array1, Array2, Array};

    // Returns MatrixSynapse of size n connected with probability p, with a set number of
    // inhibitory neurons
    pub fn from_probability(n: usize, p: f32, inhibitory: Array1<bool>) -> MatrixSynapse {
            let mut weights: Array2<f32> = random_matrix((n,n), StandardNormal)
                .mapv(|x| x.abs());

            let mut enabled_connections: Array2<f32> = random_matrix((n,n), Uniform::new(0.0, 1.0))
                .mapv(|x| if x > (1.0 - p) {1.0} else {0.0});

            // Remove connections from a neuron to itself, ie. diagonal entries.
            let e: Array2<f32> = (Array::eye(n) + 1.0) % 2.0;
            enabled_connections = &enabled_connections * &e;

            weights = weights * &enabled_connections;

            let neuron_type: Array1<f32> = inhibitory.mapv(|i| if i {-1.0} else {1.0});

            let synapses = MatrixSynapse::new(weights, neuron_type);
            log::trace!("Generated {}", synapses);
            synapses
    }
}
