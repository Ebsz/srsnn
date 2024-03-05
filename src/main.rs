mod model;
mod pools;
mod plots;
mod synapses;
mod population;
mod izhikevich;

use std::time::Instant;
use ndarray::{s, array, Array, Array1, Array2, ArrayBase};

use pools::IzhikevichPool;
use population::Population;
use izhikevich::Izhikevich;
use plots::plot_single_neuron_potential;

const N: usize = 200;
const T: usize = 1000;


fn main() {
    let input: Array2<f32> = Array::ones((T, N)) * 17.3;

    let mut pool = IzhikevichPool::matrix_pool(N);

    println!("Running..");
    let start_time = Instant::now();
    let mut pots = pool.run(T, input);

    println!("Simulated {} neurons for {} steps in {}s", N, T, (start_time.elapsed().as_secs_f32()));

    let single_pot = pots.into_iter().map(|x| x[0] as f32).collect();
    let plot_ok = plot_single_neuron_potential(single_pot);

    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),

    }
}
