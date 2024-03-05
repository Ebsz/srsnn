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
use plots::plot_single_neuron_potential;

use record::Record;

const N: usize = 2;
const T: usize = 100;


fn main() {
    let input: Array2<f32> = Array::ones((T, N)) * 17.3;

    let mut pool = IzhikevichPool::matrix_pool(N);

    let mut record = Record::new();

    println!("Running..");

    let start_time = Instant::now();
    pool.run(T, input, &mut record);

    println!("Simulated {} neurons for {} steps in {}s", N, T, (start_time.elapsed().as_secs_f32()));

    let single_pot = record.get_potentials().iter().map(|x| x[0]).collect();

    let plot_ok = plot_single_neuron_potential(single_pot);
    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }
}
