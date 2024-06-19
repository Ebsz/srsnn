pub mod izhikevich;
pub mod lapicque;

use crate::spikes::Spikes;

use ndarray::Array1;

use serde::Serialize;
use serde::de::DeserializeOwned;


/// Defines a model of a neuron
pub trait NeuronModel {
    type Parameters: Default + Copy + Clone + Serialize + DeserializeOwned;

    fn new(n: usize, params: Vec<Self::Parameters>) -> Self;
    fn step(&mut self, input: Array1<f32>) -> Spikes;
    fn reset(&mut self);
    fn potentials(&self) -> Array1<f32>;

    /// Returns an instance of the model with n neurons initialized with the default parameters
    fn n_default(n: usize) -> Self where Self: Sized {
        let default = Self::Parameters::default();

        let params: Vec<Self::Parameters> = vec![default; n];

        Self::new(n, params)
    }

    fn len(&self) -> usize {
        self.potentials().shape()[0]
    }
}


//pub struct NeuronModelDescription<N: NeuronModel> {
//    inhibitory: bool,
//    params: N::Parameters
//}
