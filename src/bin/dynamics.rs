//! luna/src/dynamics.rs

use luna::pools::Pool;
use luna::network::Network;
use luna::plots::{generate_plots, plot_network_energy};
use luna::record::{Record, RecordType, RecordDataType};
use luna::logger::init_logger;

use ndarray::{s, Array, Array1, Array2};

const N: usize = 100; // # of neurons
const T: usize = 300; // # of steps to run for
const P: f32 = 0.1;   // The probability that two arbitrary neurons are connected

const INHIBITORY_RATIO: f32 = 0.2;

const N_INPUTS: usize  = 100; // How long we provide input for
const INPUT_SIZE: f32  = 17.5;


fn last_spiketime(record: &mut Record) -> usize {
    let mut spikedata: Vec<Array1<f32>> = vec![];

    for i in record.get(RecordType::Spikes) {
        if let RecordDataType::Spikes(s) = i {
            spikedata.push(s.clone());
        } else {
            panic!("Error parsing spike records");
        }
    }

    let mut last_spiketime = 0;

    for t in 0..T {
        if spikedata[t].sum() != 0.0 {
            last_spiketime = t;
        }
    }

    last_spiketime
}

fn generate_input() -> Array2<f32> {
    let mut input: Array2<f32> = Array::zeros((T, N));
    input.slice_mut(s![..N_INPUTS, ..]).fill(INPUT_SIZE);

    input
}

//TODO: A network that can sustain activity for longer after it receives input
fn run() {
    let input = generate_input();
    let mut network = Pool::new(N, P, INHIBITORY_RATIO);

    let mut record = network.run(T, input);

    log::info!("Last spike: t={:?}", last_spiketime(&mut record));

    let energy: Vec<f32> = record.get_potentials().iter().map(|x| (x + 65.0).sum()).collect();
    plot_network_energy(energy);

    generate_plots(&record);
}

fn main() {
    init_logger();
    run();
}
