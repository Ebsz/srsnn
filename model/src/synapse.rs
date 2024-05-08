pub mod linear_synapse;
pub mod matrix_synapse;

use crate::spikes::Spikes;

use ndarray::{Array1};

pub type SynapticPotential = Array1<f32>;

/// Model of a set of synapses
pub trait Synapse {
    fn step(&mut self, input: &Spikes) -> SynapticPotential;
}
