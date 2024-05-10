pub mod izhikevich;

use crate::spikes::Spikes;

use ndarray::Array1;


/// Defines a model of a neuron
pub trait NeuronModel {
    type Parameters;

    fn step(&mut self, input: Array1<f32>) -> Spikes;
    fn reset(&mut self);

    fn potentials(&self) -> Array1<f32>;

    /// Number of neurons contained in the model
    fn len(&self) -> usize {
        self.potentials().shape()[0]
    }
}


//pub struct NeuronModelDescription<N: NeuronModel> {
//    inhibitory: bool,
//    params: N::Parameters
//}
