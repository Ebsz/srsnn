use luna::pools::IzhikevichPool;
use luna::network::Network;
use luna::plots::generate_plots;
use luna::record::Record;

use std::time::Instant;
use ndarray::{Array, Array2};


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
