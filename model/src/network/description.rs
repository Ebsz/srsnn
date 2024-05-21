//! Abstract representation of a spiking network

use crate::neuron::NeuronModel;

use ndarray::{Array1, Array2};


pub struct NetworkDescription<N: NeuronModel> {
    pub n: usize,
    pub neurons: Array1<NeuronDescription<N>>,

    pub connection_mask: Array2<u32>,
    pub weights: Array2<f32>,

    pub inputs: usize,
    pub outputs: usize,
}

impl<N: NeuronModel> NetworkDescription<N> {
    pub fn new(neurons: Array1<NeuronDescription<N>>,
        connection_mask: Array2<u32>,
        weights: Array2<f32>,
        inputs: usize,
        outputs: usize)
        -> NetworkDescription<N>
    {
        assert!(neurons.shape()[0] == connection_mask.shape()[0]);

        assert!(weights.shape()[0] == connection_mask.shape()[0]);

        NetworkDescription {
            n: neurons.shape()[0],
            neurons,
            connection_mask,
            weights,
            inputs,
            outputs
        }
    }

    // TODO: Implement
    // pub fn validate(desc: &NetworkDescription<N>) {

    // }

    // // TODO: Implement this + generic representation of connections
    // pub fn from_matrix() {

    // }
}

#[derive(Copy, Debug)]
pub struct NeuronDescription<N: NeuronModel> {
    pub id: u32,
    pub params: N::Parameters,
    pub inhibitory: bool,
    pub role: NeuronRole,
}

impl<N: NeuronModel> NeuronDescription<N> {
    pub fn new(id: u32, params: N::Parameters, inhibitory: bool, role: NeuronRole) -> NeuronDescription<N> {
        NeuronDescription {
            id,
            params,
            inhibitory,
            role
        }
    }
}

impl<N: NeuronModel> Clone for NeuronDescription<N> {
    fn clone(&self) -> NeuronDescription<N> {
        NeuronDescription {
            id: self.id,
            params: self.params,
            inhibitory: self.inhibitory,
            role: self.role
        }
    }
}


#[derive(Copy, Clone, Debug)]
pub enum NeuronRole {
    Input,
    Output,
    Network,
}
