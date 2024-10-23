pub mod representation;
pub mod basic;

use crate::spikes::Spikes;

use ndarray::Array1;


pub type SynapticPotential = Array1<f32>;

pub trait Synapse {
    fn step(&mut self, input: &Spikes) -> SynapticPotential;
    fn neuron_count(&self) -> usize;

    fn reset(&mut self);
}
