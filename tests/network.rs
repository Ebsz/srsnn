//! Tests building SpikingNetwork from NetworkRepresentation, and running the result

use model::network::builder::NetworkBuilder;
use model::neuron::NeuronModel;
use model::neuron::izhikevich::Izhikevich;
use model::network::representation::{NetworkRepresentation, NeuronDescription, NeuronRole};

use utils::random;

use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;

fn test_representation<N: NeuronModel>(n: usize, n_inputs: usize, n_outputs: usize)
-> NetworkRepresentation<NeuronDescription<N>> {
    let mut nvec: Vec<NeuronDescription<N>> = Vec::new();

    for i in 0..n_outputs {
        nvec.push(NeuronDescription::new(i as u32, Some(N::Parameters::default()), false, NeuronRole::Output));
    }

    for i in n_outputs..(n - n_inputs) {
        nvec.push(NeuronDescription::new(i as u32, Some(N::Parameters::default()), false, NeuronRole::Network));
    }

    for i in (n - n_inputs)..n {
        nvec.push(NeuronDescription::new(i as u32, None, false, NeuronRole::Input));
    }

    let neurons = Array::from_vec(nvec);

    let connection_mask: Array2<u32> = random::random_matrix((n, n), Uniform::new(0, 2));

    //let connection_mask: Array2<u32> = array![[0,1, 0],[0, 0, 1], [1,0,0]];
    let weights: Array2<f32> = connection_mask.map(|x| *x as f32);

    NetworkRepresentation::new(neurons, connection_mask, weights, n_inputs, n_outputs)
}


#[test]
fn can_build_network_from_representation() {
    let desc = test_representation::<Izhikevich>(100, 10,10);

    let _network = NetworkBuilder::build(&desc);
}
