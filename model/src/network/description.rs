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
    fn new(neurons: Array1<NeuronDescription<N>>,
        connection_mask: Array2<u32>,
        weights: Array2<f32>,
        inputs: usize,
        outputs: usize)
        -> NetworkDescription<N>
    {
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

pub struct NeuronDescription<N: NeuronModel> {
    pub params: N::Parameters,
    pub inhibitory: bool,
    pub role: NeuronRole,
}

pub enum NeuronRole {
    Input,
    Output,
    Network,
}
