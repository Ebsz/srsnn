
use srsnn::plots;

use model::network::{Network, SpikingNetwork};
use model::neuron::NeuronModel;
use model::neuron::izhikevich::{Izhikevich, IzhikevichParameters};

use model::synapse::Synapse;
use model::synapse::basic::BasicSynapse;
use model::synapse::representation::MatrixRepresentation;
use model::synapse::exponential::ExponentialSynapse;
use model::synapse::bi_exponential::BiExponentialSynapse;

use utils::environment::Environment;

use ndarray::{array, Array, Array1};


const N: usize = 1;
const T: u32 = 300;

const SPIKE_FREQ: u32 = 10;

const INPUT_WEIGHT: f32 = 5.0;

pub fn init() {
    single_neuron_network();
}



pub fn single_neuron_network() {
    let neurons = Izhikevich::n_default(N);
    println!("Parameters: {:?}", IzhikevichParameters::default());

    //let neurons = Izhikevich::new(1, vec![ IzhikevichParameters { a: 0.02, b: 0.2, c: -50.0, d: 2.0 }]);

    let weights = array!([0.0]);
    let ntype = array![1.0];
    let input_matrix = Array::ones((1,1)) * INPUT_WEIGHT;

    println!("wt: {}", INPUT_WEIGHT);

    // BasicSynapse
    //let representation = MatrixRepresentation::new(weights, ntype);
    //let input_representation = MatrixRepresentation::new(input_matrix, array![1.0]);

    //let input_synapse = BasicSynapse::new(input_representation);
    //let synapse = BasicSynapse::new(representation);

    // ExponentialSynapse
    let synapse = ExponentialSynapse::new(weights, ntype);
    let input_synapse = ExponentialSynapse::new(input_matrix, array![1.0]);

    // BiExponentialSynapse
    //let synapse = BiExponentialSynapse::new(weights, ntype);
    //let input_synapse = BiExponentialSynapse::new(input_matrix, array![1.0]);

    let env = Environment { inputs: 1, outputs: 0, };


    let mut network = SpikingNetwork::new(neurons, synapse, input_synapse, env);
    network.recording = true;

    run_network(network);
}

fn run_network<N: NeuronModel, S: Synapse>(mut network: SpikingNetwork<N, S>) {
    log::info!("Running network");

    for t in 0..T {
        let input = input(t);

        let out = network.step(input.into());
    }

    plots::single_neuron_dynamics(&network.record);

    //plots::plot_run_spikes(&network.record, None);
    //plots::generate_plots(&network.record);
}

fn input(t: u32) -> Array1<u32> {
    if t % SPIKE_FREQ == 0 {
        array![1]
    } else  {
        array![0]
    }
}

pub fn run_single_neuron() {
    let mut neuron = Izhikevich::n_default(1);

    let mut potentials: Array1<f32> = Array::zeros(T as usize);

    for t in 0..T {
        let input = array![10.0];
        neuron.step(input);

        potentials[t as usize] = neuron.potentials()[0];
    }

    plots::plot_single_neuron_potential(potentials.as_slice().unwrap());
}
