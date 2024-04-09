//! A pool is a randomly generated network shaped by certain parameters

use crate::network::Network;
use crate::model::neuron::NeuronModel;
use crate::model::neuron::izhikevich::Izhikevich;
use crate::model::synapse::Synapse;
use crate::model::synapse::matrix_synapse::MatrixSynapse;
use crate::model::synapse::linear_synapse::LinearSynapse;

use crate::gen::synapse_gen;

use ndarray::{s, Array, Array1};


pub struct Pool<S: Synapse> {
    neurons: Izhikevich,
    synapse: S
}

impl<S: Synapse> Network<Izhikevich, S> for Pool<S> {
    fn model(&mut self) -> &mut dyn NeuronModel {
        &mut self.neurons
    }

    fn synapse(&mut self) -> &mut dyn Synapse {
        &mut self.synapse
    }
}

impl Pool<MatrixSynapse> {
    pub fn new(n: usize, p: f32, inhibitory_fraction: f32) -> Pool<MatrixSynapse> {
        let model = Izhikevich::default(n);

        let k = (n as f32 * inhibitory_fraction ) as usize;

        let mut inhibitory: Array1<bool> = Array::zeros(n).mapv(|_: f32| false);
        inhibitory.slice_mut(s![..k]).fill(true);

        let synapse = synapse_gen::from_probability(n, p, inhibitory);

        Pool {
            neurons: model,
            synapse
        }
    }
}

impl Pool<LinearSynapse> {
    pub fn linear_pool(n: usize, p: f32) -> Pool<LinearSynapse> {
        let izh = Izhikevich::default(n);
        let synapse = LinearSynapse::from_probability(n, p);

        Pool {
            neurons: izh,
            synapse
        }
    }
}
