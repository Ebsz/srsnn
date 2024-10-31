pub mod representation;
pub mod basic;
pub mod exponential;
pub mod bi_exponential;

use crate::spikes::Spikes;

use ndarray::Array1;


pub type SynapticPotential = Array1<f32>;

pub trait Synapse {
    fn step(&mut self, input: &Spikes) -> SynapticPotential;

    fn shape(&self) -> (usize, usize);

    fn reset(&mut self);
}
