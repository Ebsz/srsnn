pub mod representation;
pub mod basic;
pub mod exponential;
pub mod bi_exponential;

use crate::spikes::Spikes;

use ndarray::{Array1, Array2};

/// A vector describing the neuron type of N neurons, where entry i is -1.0 if
/// neuron i is inhibitory, or 1.0 if it is excitatory.
pub type NeuronType = Array1<f32>;

pub type SynapticPotential = Array1<f32>;

pub trait Synapse {
    fn new(w: Array2<f32>, neuron_type: Array1<f32>) -> Self;
    fn step(&mut self, input: &Spikes) -> SynapticPotential;

    fn shape(&self) -> (usize, usize);

    fn reset(&mut self);
}
