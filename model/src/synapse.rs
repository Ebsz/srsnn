pub mod matrix_synapse;
pub mod representation;

use crate::spikes::Spikes;
use crate::synapse::representation::{SynapseRepresentation, MapRepresentation, MatrixRepresentation};

use ndarray::{Array1, Array2};


pub type SynapticPotential = Array1<f32>;

pub trait Synapse {
    fn step(&mut self, input: &Spikes) -> SynapticPotential;
    fn neuron_count(&self) -> usize;
}

pub struct BaseSynapse<R: SynapseRepresentation> {
    pub representation: R
}

impl<R: SynapseRepresentation> Synapse for BaseSynapse<R> {
    fn step(&mut self, input: &Spikes) -> SynapticPotential {
        self.representation.step(input)
    }

    fn neuron_count(&self) -> usize{
        self.representation.neuron_count()
    }
}

impl<R: SynapseRepresentation> BaseSynapse<R> {
    pub fn new(representation: R) -> BaseSynapse<R> {
        BaseSynapse {
            representation
        }
    }

    pub fn from_matrix(weights: Array2<f32>, neuron_type: Array1<f32>) -> BaseSynapse<MatrixRepresentation> {
        BaseSynapse {
            representation: MatrixRepresentation::new(weights, neuron_type)
        }
    }
}

impl From<&BaseSynapse<MatrixRepresentation>> for BaseSynapse<MapRepresentation> {
    fn from(item: &BaseSynapse<MatrixRepresentation>) -> BaseSynapse<MapRepresentation> {
        BaseSynapse::<MapRepresentation>::new(MapRepresentation::from(&item.representation))
    }
}

#[test]
fn test_create_and_step_base_synapse() {
    let a = ndarray::array![[1.0, 2.0],[3.0,4.0]];

    let mut m: BaseSynapse<MatrixRepresentation> =
        BaseSynapse::<MatrixRepresentation>::from_matrix(a, ndarray::Array::zeros(2));

    let mut input = Spikes::new(2);
    let k = m.step(&input);
}
