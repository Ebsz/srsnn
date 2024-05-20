//! A pool is a randomly generated network shaped by certain parameters

use model::neuron::NeuronModel;
use model::neuron::izhikevich::Izhikevich;
use model::synapse::{BaseSynapse, Synapse};
use model::synapse::representation::{MapRepresentation, MatrixRepresentation};
use model::network::runnable::RunnableNetwork;

use crate::gen::synapse_gen;

use ndarray::{s, Array, Array1};


pub struct Pool<S: Synapse> {
    neurons: Izhikevich,
    synapse: S
}

impl<S: Synapse> RunnableNetwork<Izhikevich, S> for Pool<S> {
    fn model(&mut self) -> &mut Izhikevich {
        &mut self.neurons
    }

    fn synapse(&mut self) -> &mut S {
        &mut self.synapse
    }
}

impl Pool<BaseSynapse<MatrixRepresentation>> {
    pub fn new(n: usize, p: f32, inhibitory_fraction: f32) -> Pool<BaseSynapse<MatrixRepresentation>> {
        let model = Izhikevich::n_default(n);

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

impl Pool<BaseSynapse<MapRepresentation>> {
    pub fn linear_pool(n: usize, p: f32) -> Pool<BaseSynapse<MapRepresentation>> {
        let izh = Izhikevich::n_default(n);

        // TODO: add inhibitory_fraction param
        let inhibitory = Array::zeros(n).mapv(|_: u32| false);

        let synapse = synapse_gen::linear_from_probability(n, p, inhibitory);

        Pool {
            neurons: izh,
            synapse
        }
    }
}
