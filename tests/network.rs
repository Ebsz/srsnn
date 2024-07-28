//! Tests building SpikingNetwork from NetworkRepresentation, and running the result

use model::network::builder::NetworkBuilder;
use model::neuron::NeuronModel;
use model::neuron::izhikevich::{Izhikevich, IzhikevichParameters};
use model::network::representation::{NetworkRepresentation, NeuronDescription};

use utils::random;
use utils::environment::Environment;

use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;

fn test_representation<N: NeuronModel>(n: usize, env: Environment)
-> NetworkRepresentation<NeuronDescription<N>> {
    let mut nvec: Vec<NeuronDescription<N>> = Vec::new();

    for i in 0..n {
        nvec.push(NeuronDescription::new(
                i as u32,
                N::Parameters::default(),
                false,
        ));

    }

    let neurons = Array::from_vec(nvec);

    let connection_mask: Array2<u32> = random::random_matrix((n, n), Uniform::new(0, 2));

    let weights: Array2<f32> = connection_mask.map(|x| *x as f32);

    let mut input_cm: Array2<u32> = Array::zeros((n, env.inputs));

    for i in 0..env.inputs {
        input_cm[[i,i]] = 1;
    }

    let input_w: Array2<f32> = Array::ones((n, env.inputs));

    NetworkRepresentation::new(neurons, connection_mask, weights, input_cm, input_w, env)
}


#[test]
fn can_build_network_from_representation() {
    let env = Environment {
        inputs: 10,
        outputs: 10,
    };

    let desc = test_representation::<Izhikevich>(100, env);

    let _network = NetworkBuilder::build(&desc);
}
