//! A pool is a group of randomly generated neurons shaped by certain parameters

use crate::model::NeuronModel;
use crate::model::izhikevich::Izhikevich;
use crate::synapses::{Synapses, MatrixSynapses, LinearSynapses};

use crate::network::Network;

use ndarray::{Array2, Array};


/// A pool consisting of Izhikevich neurons connected by a certain type of synapse
pub struct IzhikevichPool<S: Synapses> {
    neurons: Izhikevich,
    synapses: S 
}

impl<S: Synapses> Network<Izhikevich, S> for IzhikevichPool<S> {
    fn model(&mut self) -> &mut dyn NeuronModel {
        &mut self.neurons
    }

    fn synapses(&mut self) -> &mut dyn Synapses {
        &mut self.synapses
    }
}

impl IzhikevichPool<MatrixSynapses> {
    pub fn matrix_pool(n: usize) -> IzhikevichPool<MatrixSynapses> {
        let izh = Izhikevich::default(n);
        let synapses = MatrixSynapses::new(n);

        IzhikevichPool {
            neurons: izh,
            synapses
        }
    }
}

impl IzhikevichPool<LinearSynapses> {
    pub fn linear_pool(n: usize, p: f32) -> IzhikevichPool<LinearSynapses> {
        let izh = Izhikevich::default(n);
        let synapses = LinearSynapses::from_probability(n, p);

        IzhikevichPool {
            neurons: izh,
            synapses
        }
    }
}
