//! gen.rs
//!
//! Functions for randomly generating models
//!

use utils::random;

use ndarray::{Array, Array1, Array2, Zip};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};


pub mod stochastic {
    use super::*;

    /// A connection probability matrix defines a probability over each edge in the network
    pub type ConnectionProbabilityMatrix = Array2<f32>;

    pub fn sample_connection_probability_matrix(pm: &ConnectionProbabilityMatrix) -> Array2<u32> {
        let n = pm.shape()[0];
        let mut connection_mask: Array2<u32> = Array::zeros((n,n));

        let samples: Array2<f32> = random::random_matrix((n,n), Uniform::new(0.0, 1.0));

        Zip::from(&mut connection_mask)
            .and(pm)
            .and(&samples)
            .for_each(|c, p, s| *c = if s <= p { 1 } else { 0 });

        connection_mask
    }
}


pub mod synapse_gen {
    use super::*;

    use model::synapse::BaseSynapse;
    use model::synapse::representation::{MatrixRepresentation, MapRepresentation};


    // Returns a synapse of size n connected with probability p, with a set number of
    // inhibitory neurons
    pub fn from_probability(n: usize, p: f32, inhibitory: Array1<bool>) -> BaseSynapse<MatrixRepresentation> {
        let mut weights: Array2<f32> = random::random_matrix((n,n), StandardNormal)
            .mapv(|x: f32| x.abs());

        let mut enabled_connections: Array2<f32> = random::random_matrix((n,n), Uniform::new(0.0, 1.0))
            .mapv(|x| if x > (1.0 - p) {1.0} else {0.0});

        // Remove connections from a neuron to itself, ie. diagonal entries.
        let e: Array2<f32> = (Array::eye(n) + 1.0) % 2.0;
        enabled_connections = &enabled_connections * &e;

        weights = weights * &enabled_connections;

        let neuron_type: Array1<f32> = inhibitory.mapv(|i| if i {-1.0} else {1.0});

        let representation = MatrixRepresentation::new(weights, neuron_type);

        BaseSynapse::new(representation)
    }

    pub fn linear_from_probability(n: usize, p: f32, inhibitory: Array1<bool>) -> BaseSynapse<MapRepresentation> {
        let m = from_probability(n, p, inhibitory);

        BaseSynapse::new(MapRepresentation::from(&m.representation))
    }
}

#[test]
fn test_sampling_connection_probability_matrix() {
    let n = 10;

    let fully_connected = Array2::<u32>::ones((n,n));
    let no_connections = Array2::<u32>::zeros((n,n));

    let p1 = Array::ones((n,n));
    let c1 = stochastic::sample_connection_probability_matrix(&p1);

    assert!(c1 == fully_connected);

    let p2 = Array::zeros((n,n));
    let c2 = stochastic::sample_connection_probability_matrix(&p2);

    assert!(c2 == no_connections);
}
