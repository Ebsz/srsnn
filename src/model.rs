pub mod izhikevich;

use ndarray::{Array, Array1};


pub struct Spikes {
    pub data: Array1<f32>
}

impl Spikes {
    pub fn new(n: usize) -> Spikes {
        Spikes {
            data: Array::zeros(n)
        }
    }

    /// Get the indices of neurons that fire
    pub fn firing(&self) -> Vec<usize> {
        self.data.iter().enumerate().filter(|(_, n)| **n != 0.0).map(|(i,_)| i).collect()
    }

    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }
}


/// Defines a model of a neuron
pub trait NeuronModel { 
    fn step(&mut self, input: Array1<f32>) -> Spikes;

    fn potentials(&self) -> Array1<f32>;

    /// Number of neurons contained in the model
    fn size(&self) -> usize {
        self.potentials().shape()[0]
    }
}
