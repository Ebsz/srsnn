use ndarray::{Array, Array1};

pub struct FiringState {
    pub state: Array1<f32>
}

impl FiringState {
    pub fn new(n: usize) -> FiringState {
        FiringState {
            state: Array::zeros(n)
        }
    }
}


/// Defines a model of a neuron
pub trait NeuronModel { 
    fn step(&mut self, input: Array1<f32>) -> FiringState;

    fn potentials(&self) -> Array1<f32>;
}
