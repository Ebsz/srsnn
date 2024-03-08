pub mod izhikevich;

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

    /// Get the indices of neurons that fire
    pub fn firing(&self) -> Vec<usize> {
        self.state.iter().enumerate().filter(|(_, n)| **n != 0.0).map(|(i,_)| i).collect()
    }

    pub fn len(&self) -> usize {
        self.state.shape()[0]
    }
}


/// Defines a model of a neuron
pub trait NeuronModel { 
    fn step(&mut self, input: Array1<f32>) -> FiringState;

    fn potentials(&self) -> Array1<f32>;
}
