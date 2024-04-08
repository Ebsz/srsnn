pub mod linear_synapse;
pub mod matrix_synapse;

use crate::model::spikes::Spikes;

use ndarray::{Array1};


/// Model of a set of synapses
pub trait Synapse {
    fn step(&mut self, input: &Spikes) -> Array1<f32>;
}
