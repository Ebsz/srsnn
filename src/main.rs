mod model;
mod pools;
mod plots;
mod record;
mod synapses;
mod population;
mod izhikevich;

use std::time::Instant;
use ndarray::{Array, Array2};

use pools::IzhikevichPool;
use population::Population;
use plots::generate_plots;

use record::Record;

const N: usize = 100; // # of neuron
const T: usize = 300; // # of steps to run for
const P: f32 = 0.1;   // The probability that two arbitrary neurons are connected


fn main() {
    let input: Array2<f32> = Array::ones((T, N)) * 17.3;

    let mut pool = IzhikevichPool::linear_pool(N, P);

    let mut record = Record::new();

    println!("Running..");

    let start_time = Instant::now();
    pool.run(T, input, &mut record);

    println!("Simulated {} neurons for {} steps in {}s", N, T, (start_time.elapsed().as_secs_f32()));

    generate_plots(&record);
}
