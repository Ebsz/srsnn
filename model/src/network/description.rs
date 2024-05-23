//! Abstract representation of a spiking network

use crate::neuron::NeuronModel;

use ndarray::{Array1, Array2};


pub struct NetworkDescription<N> {
    pub n: usize,
    pub neurons: Array1<N>,

    pub connection_mask: Array2<u32>,
    pub weights: Array2<f32>,

    pub inputs: usize,
    pub outputs: usize,
}

impl<N> NetworkDescription<N> {
    pub fn new(neurons: Array1<N>,
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
}

#[derive(Copy, Debug)]
pub struct NeuronDescription<N: NeuronModel> {
    pub id: u32,
    pub params: Option<N::Parameters>,
    pub inhibitory: bool,
    pub role: NeuronRole,
}

impl<N: NeuronModel> NeuronDescription<N> {
    pub fn new(id: u32, params: Option<N::Parameters>, inhibitory: bool, role: NeuronRole) -> NeuronDescription<N> {
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


#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NeuronRole {
    Input,
    Output,
    Network,
}
