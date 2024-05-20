use crate::neuron::NeuronModel;
use crate::synapse::Synapse;

use std::collections::HashMap;
use ndarray::Array2;


pub struct NeuronDescription<N: NeuronModel> {
    params: N::Parameters,
    inhibitory: bool,
    ntype: NeuronType,
}

pub struct Connections {

}

pub enum NeuronType {
    Input,
    Output,
    Network,
}


pub struct NetworkDescription<N: NeuronModel> {
    neurons: HashMap<u32, NeuronDescription<N>>,
    connections: Array2<u32>,
    inputs: usize,
    outputs: usize,
}

//impl NetworkDescription<N> {
//    fn new(neurons: Vec<NeuronDescription>, connections: Connections
//
//}
