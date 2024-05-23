//! Creates a runnable instance from an abstract representation.
//!
//! Builds a SpikingNetwork from a NetworkDescription,

use crate::network::SpikingNetwork;
use crate::network::description::{NetworkDescription, NeuronDescription, NeuronRole};
use crate::neuron::NeuronModel;
use crate::synapse::BaseSynapse;

use crate::synapse::representation::{NeuronType, MatrixRepresentation};

use ndarray::{Array, Array1};


pub struct NetworkBuilder;

impl NetworkBuilder {
    pub fn build<N: NeuronModel>(desc: NetworkDescription<NeuronDescription<N>>)
        -> SpikingNetwork<N, BaseSynapse<MatrixRepresentation>> {
        let neuron_params = Self::parse_neuron_params(&desc);
        let neuron_types = Self::parse_neuron_types(&desc);

        let n_network_neurons = desc.n - desc.inputs;

        let model = N::new(n_network_neurons, neuron_params);

        let synapse_matrix = &(desc.connection_mask.mapv(|v| v as f32)) * &desc.weights;
        let representation = MatrixRepresentation::new(synapse_matrix, neuron_types);
        let synapse = BaseSynapse::new(representation);

        SpikingNetwork::new(model, synapse, desc.inputs, desc.outputs)
    }

    fn parse_neuron_types<N: NeuronModel> (desc: &NetworkDescription<NeuronDescription<N>>) -> NeuronType {
        desc.neurons.map(|n| if n.inhibitory {-1.0} else { 1.0 })
    }

    fn parse_neuron_params<N: NeuronModel> (desc: &NetworkDescription<NeuronDescription<N>>) -> Vec<N::Parameters> {
        desc.neurons.iter().filter_map(|n| n.params).collect()
    }
}
