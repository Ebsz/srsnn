use crate::spikes::Spikes;
use crate::synapse::{Synapse, SynapticPotential};
use crate::synapse::representation::{SynapseRepresentation, MapRepresentation, MatrixRepresentation};

use ndarray::{Array1, Array2};


pub struct BasicSynapse<R: SynapseRepresentation> {
    pub representation: R
}

impl<R: SynapseRepresentation> Synapse for BasicSynapse<R> {
    fn step(&mut self, input: &Spikes) -> SynapticPotential {
        self.representation.step(input)
    }

    fn shape(&self) -> (usize, usize) {
        self.representation.shape()
    }

    fn reset(&mut self) {
        // this does nothing, because the base synapse does not have state
    }
}

impl<R: SynapseRepresentation> BasicSynapse<R> {
    pub fn new(representation: R) -> BasicSynapse<R> {
        BasicSynapse {
            representation
        }
    }

    pub fn from_matrix(weights: Array2<f32>, neuron_type: Array1<f32>) -> BasicSynapse<MatrixRepresentation> {
        BasicSynapse {
            representation: MatrixRepresentation::new(weights, neuron_type)
        }
    }
}

impl From<&BasicSynapse<MatrixRepresentation>> for BasicSynapse<MapRepresentation> {
    fn from(item: &BasicSynapse<MatrixRepresentation>) -> BasicSynapse<MapRepresentation> {
        BasicSynapse::<MapRepresentation>::new(MapRepresentation::from(&item.representation))
    }
}

#[test]
fn test_create_and_step_base_synapse() {
    let m = ndarray::array![[1.0, 2.0],[3.0,4.0]];
    let input = Spikes::new(2);

    let mut a: BasicSynapse<MatrixRepresentation> =
        BasicSynapse::<MatrixRepresentation>::from_matrix(m, ndarray::Array::zeros(2));

    let a_out = a.step(&input);

    let mut b: BasicSynapse<MapRepresentation> =
        BasicSynapse::<MapRepresentation>::from(&a);

    let b_out = b.step(&input);

    assert_eq!(a_out, b_out);
}
