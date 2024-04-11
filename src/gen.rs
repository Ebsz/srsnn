//! gen.rs
//!
//! Functions for randomly generating models


pub mod synapse_gen {
    use model::synapse::matrix_synapse::MatrixSynapse;
    use model::synapse::linear_synapse::LinearSynapse;

    use utils::random::{random_range, random_sample, random_matrix};

    use ndarray_rand::rand_distr::{StandardNormal, Uniform};
    use ndarray::{Array1, Array2, Array};

    use std::collections::HashMap;

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

    pub fn linear_from_probability(n: usize, p: f32) -> LinearSynapse {
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

        LinearSynapse::new(connections)
    }
}
