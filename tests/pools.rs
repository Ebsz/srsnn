use luna::pools::Pool;
use luna::network::Network;
use luna::record::Record;

use ndarray::{Array, Array2};


const N: usize = 100; // # of neurons
const T: usize = 1000; // # of steps to run for
const P: f32 = 0.1;   // The probability that two arbitrary neurons are connected

#[test]
fn matrix_pool_can_run_and_has_positive_energy() {
    let input: Array2<f32> = Array::ones((T, N)) * 17.3;
    let mut network = Pool::new(N, P, 0.2);

    let mut record = network.run(T, &input);
    let energy: Vec<f32> = record.get_potentials().iter().map(|x| (x + 65.0).sum()).collect();
    assert!(energy.iter().sum::<f32>() > 0.0);
}

#[test]
fn linear_pool_can_run_and_has_positive_energy() {
    let input: Array2<f32> = Array::ones((T, N)) * 17.3;
    let mut network = Pool::linear_pool(N, P);

    let mut record = network.run(T, &input);

    let energy: Vec<f32> = record.get_potentials().iter().map(|x| (x + 65.0).sum()).collect();

    assert!(energy.iter().sum::<f32>() > 0.0);
}
