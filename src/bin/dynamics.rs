//! luna/src/dynamics.rs

use luna::pools::Pool;
use luna::visual::plots::{generate_plots, plot_network_energy};

use model::network::runnable::RunnableNetwork;
use model::record::{Record, RecordType, RecordDataType};

use utils::logger::init_logger;

use ndarray::{s, Array, Array1, Array2};
use std::collections::HashMap;

const N: usize = 100; // # of neurons
const T: usize = 1000; // # of steps to run for
const P: f32 = 0.1;   // The probability that two arbitrary neurons are connected

const INHIBITORY_RATIO: f32 = 0.2;

const INPUT_T: usize  = 100; // How long we provide input for

const DEFAULT_INPUT_SIZE: f32  = 17.5;


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

fn generate_network_input(n: usize, input_size: f32) -> Array2<f32> {
    let mut input: Array2<f32> = Array::zeros((T, n));
    input.slice_mut(s![..INPUT_T, ..]).fill(input_size);

    input
}

fn plot(record: &Record) {
    let energy: Vec<f32> = record.get_potentials().iter().map(|x| (x + 65.0).sum()).collect();
    let _ = plot_network_energy(energy);

    generate_plots(record);
}

//fn plot_psth(record: &Record) {
//    let potentials: Vec<Array1<f32>> = record.get_potentials();
//
//    let spikedata: Array2<f32> = Array::zeros((potentials.len(), potentials[0].shape()[0]));
//
//    for (i, p) in potentials.iter().enumerate() {
//        println!("{:?}, {:?}", p, i);
//    }
//
//    let psth = luna::analysis::to_firing_rate(spikedata);
//}

/// Iterates a network over a set of network parameters
/// In a sense, we compute a function from parameter space to state space * time
fn iterate() {
    log::info!("Iterating");

    //let inputs: Array1<f32> = Array::range(15.,20.0,0.1);

    let network_sizes: Array1<f32> = Array::range(100., 160., 2.);

    let mut spiketimes: Vec<usize> = Vec::new();

    let mut records: HashMap<usize, Record> = HashMap::new();

    for n in network_sizes.iter() {
        let network_input = generate_network_input(*n as usize, 17.5);

        let mut network = Pool::new(*n as usize, P, INHIBITORY_RATIO);
        let mut record = network.run(T, &network_input);

        let spiketime = last_spiketime(&mut record);
        log::info!("n: {n}, spiketime: {}", spiketime);
        spiketimes.push(spiketime);
        records.insert(*n as usize, record);
    }

    if let Some(r) = records.get(&142) {
        plot(&r);
    }
}


#[allow(dead_code)]
fn run() {
    log::info!("Running network");
    let input = generate_network_input(N, DEFAULT_INPUT_SIZE);
    let mut network = Pool::new(N, P, INHIBITORY_RATIO);

    let mut record = network.run(T, &input);

    log::info!("Last spike: t={:?}", last_spiketime(&mut record));

    let _ = plot(&record);
}

fn main() {
    init_logger(None);
    //iterate();
    run();
}
