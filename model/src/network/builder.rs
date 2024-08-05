//! Creates a runnable instance from an abstract representation.
//!
//! Builds a SpikingNetwork from a NetworkRepresentation,

use crate::network::SpikingNetwork;
use crate::network::representation::{NetworkRepresentation, NeuronDescription};
use crate::neuron::NeuronModel;
use crate::synapse::BaseSynapse;

use crate::synapse::representation::{NeuronType, MatrixRepresentation};


pub struct NetworkBuilder;

impl NetworkBuilder {
    pub fn build<N: NeuronModel>(desc: &NetworkRepresentation<NeuronDescription<N>>)
        -> SpikingNetwork<N, BaseSynapse<MatrixRepresentation>> {
        let neuron_params = Self::parse_neuron_params(desc);
        let neuron_types = Self::parse_neuron_types(desc);

        let model = N::new(desc.n, neuron_params);

        let synapse_matrix = &(desc.network_cm.mapv(|v| v as f32)) * &desc.network_w;

        let input_matrix = &(desc.input_cm.mapv(|v| v as f32)) * &desc.input_w;
        let output_matrix = desc.output_cm.clone();

        let representation = MatrixRepresentation::new(synapse_matrix, neuron_types);
        let synapse = BaseSynapse::new(representation);

        SpikingNetwork::new(model, synapse, input_matrix, output_matrix, desc.env.clone())
    }

    fn parse_neuron_types<N: NeuronModel> (desc: &NetworkRepresentation<NeuronDescription<N>>) -> NeuronType {
        desc.neurons.map(|n| if n.inhibitory {-1.0} else { 1.0 })
    }

    fn parse_neuron_params<N: NeuronModel> (desc: &NetworkRepresentation<NeuronDescription<N>>) -> Vec<N::Parameters> {
        desc.neurons.iter().map(|n| n.params).collect()
    }
}
