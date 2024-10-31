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

pub fn run_network<N: NeuronModel, S: Synapse, F: Fn(u32) -> (Array1<u32>)>(
    mut network: SpikingNetwork<N, S>,
    input: F)
{
    log::info!("Running network");

    for t in 0..T {
        let input = input(t);

        let out = network.step(input.into());
    }

    plots::single_neuron_dynamics(&network.record);

    //plots::plot_run_spikes(&network.record, None);
    //plots::generate_plots(&network.record);
}

fn random_network(n: usize) {
    let a = csa::mask::random(0.3).matrix(8).map(|x| *x as f32);

    println!("{}", a);
}
