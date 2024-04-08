pub mod izhikevich;

use ndarray::Array1;

use crate::model::spikes::Spikes;


/// Defines a model of a neuron
pub trait NeuronModel {
    fn step(&mut self, input: Array1<f32>) -> Spikes;

    fn potentials(&self) -> Array1<f32>;

    /// Number of neurons contained in the model
    fn size(&self) -> usize {
        self.potentials().shape()[0]
    }
}
