//! Creates a runnable instance from an abstract representation.
//!
//! Builds a SpikingNetwork from a NetworkRepresentation,

use crate::network::SpikingNetwork;
use crate::network::representation::{NetworkRepresentation, NeuronDescription};
use crate::neuron::NeuronModel;
use crate::synapse::{Synapse, NeuronType};

use ndarray::{s, Array};


pub struct NetworkBuilder;

impl NetworkBuilder {
    pub fn build<N: NeuronModel, S: Synapse>(desc: &NetworkRepresentation<NeuronDescription<N>>)
        -> SpikingNetwork<N, S> {
        let neuron_params = Self::parse_neuron_params(desc);
        let neuron_types = Self::parse_neuron_types(desc);

        // ensure output neurons are excitatory
        assert!(neuron_types.slice(s![-(desc.env.outputs as i32)..]).iter().all(|x| *x == 1.0));

        let model = N::new(desc.n, neuron_params);

        let synapse_matrix = &(desc.network_cm.mapv(|v| v as f32)) * &desc.network_w;
        let synapse = S::new(synapse_matrix, neuron_types);

        let input_matrix = &(desc.input_cm.mapv(|v| v as f32)) * &desc.input_w;
        let input_synapse = S::new(input_matrix, Array::ones(desc.env.inputs));

        //use crate::network::representation::DefaultRepresentation;
        //utils::data::save::<NetworkRepresentation<NeuronDescription<N>>>(desc.clone(), "network.json");
        //panic!("saved");

        SpikingNetwork::new(model, synapse, input_synapse, desc.env.clone())
    }

    fn parse_neuron_types<N: NeuronModel> (desc: &NetworkRepresentation<NeuronDescription<N>>) -> NeuronType {
        desc.neurons.map(|n| if n.inhibitory {-1.0} else { 1.0 })
    }

    fn parse_neuron_params<N: NeuronModel> (desc: &NetworkRepresentation<NeuronDescription<N>>) -> Vec<N::Parameters> {
        desc.neurons.iter().map(|n| n.params).collect()
    }
}
