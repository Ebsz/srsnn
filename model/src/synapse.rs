pub mod linear_synapse;
pub mod matrix_synapse;
pub mod representation;

use crate::spikes::Spikes;
use crate::synapse::representation::{Representation, SynapseRepresentation, MapRepresentation, MatrixRepresentation};

use ndarray::{Array1, Array2};


pub type SynapticPotential = Array1<f32>;

pub trait Synapse {
    fn step(&mut self, input: &Spikes) -> SynapticPotential;
}


pub struct BaseSynapse {
    representation: Representation
}

impl Synapse for BaseSynapse {
    fn step(&mut self, input: &Spikes) -> SynapticPotential {
        self.representation.step(input)
    }
}


impl BaseSynapse {
    pub fn new(representation: Representation) -> BaseSynapse {
        BaseSynapse {
            representation
        }
    }

    pub fn matrix_repr(weights: Array2<f32>, neuron_type: Array1<f32>) -> BaseSynapse {
        BaseSynapse {
            representation: Representation::Matrix(MatrixRepresentation::new(weights, neuron_type))
        }
    }

    fn to_map_representation(&mut self) {
        self.representation = match &self.representation {
            Representation::Matrix(m) => { Representation::Map(MapRepresentation::from(m)) },
            Representation::Map(_) => {panic!("Synapse is already  map representation"); },
            _ => {panic!("Could not convert synapse to map representation");}
        };
    }
}
