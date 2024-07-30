/// Generic model of a recurrent spiking neural network

use csa::{ConnectionSet, ValueSet, NeuronSet, NeuralSet};

use model::Model;
use model::network::representation::{DefaultRepresentation, NetworkRepresentation, NeuronDescription};
use model::neuron::izhikevich::IzhikevichParameters;

use utils::parameters::ParameterSet;
use utils::config::{Configurable, ConfigSection};
use utils::environment::Environment;

use serde::Deserialize;

use ndarray::{array, Array, Array2};

use std::sync::Arc;
use std::fmt::Debug;


pub trait RSNN: Configurable + Clone + Debug + Sync {
    fn get(p: &ParameterSet, config: &RSNNConfig<Self>) -> NeuralSet;
    fn params(config: &RSNNConfig<Self>) -> ParameterSet;

    fn default_dynamics() -> NeuronSet {
        NeuronSet { f: Arc::new(
            move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        )}
    }

    fn default_weights(n: usize) -> ValueSet {
        ValueSet { f: Arc::new(
            move |_i, _j| 1.0
        )}
    }

    /// 4 Izhikevich parameters + inhibitory flag
    const N_DYNAMICAL_PARAMETERS: usize = 5;
}

pub struct RSNNModel<R: RSNN> {
    n: usize,
    conf: RSNNConfig<R>,

    params: ParameterSet,
    env: Environment,
}

impl<R: RSNN> Model for RSNNModel<R> {
    fn new(
        conf: &RSNNConfig<R>,
        params: &ParameterSet,
        env: &Environment) -> Self {
        RSNNModel {
            n: conf.n,
            params: params.clone(),
            env: env.clone(),

            conf: conf.clone()
        }
    }

    fn develop(&self) -> DefaultRepresentation {
        let neural_set = R::get(&self.params, &self.conf);

        let mask = neural_set.m;
        let dynamics = &neural_set.d[0];

        // Self connections are always removed
        let network_cm = (mask - csa::mask::one_to_one()).matrix(self.n);

        let d = dynamics.vec(self.n);

        let mut neurons = Vec::new();

        for i in 0..self.n {
            let inhibitory = if d[i][4] == 1.0 { true } else { false };
            neurons.push(NeuronDescription::new(
                    i as u32,
                    IzhikevichParameters {
                        a: d[i][0],
                        b: d[i][1],
                        c: d[i][2],
                        d: d[i][3],
                    },
                    inhibitory,
            ));
        }

        let network_w = neural_set.v[0].matrix(self.n);

        let mut input_cm: Array2<u32> = Array::zeros((self.n, self.env.inputs));

        for i in 0..self.env.inputs {
            input_cm[[i,i]] = 1;
        }

        let input_w: Array2<f32> = Array::ones((self.n, self.env.inputs));

        NetworkRepresentation::new(neurons.into(), network_cm, network_w, input_cm, input_w, self.env.clone())
    }

    fn params(config: &RSNNConfig<R>) -> ParameterSet {
        R::params(config)
    }
}

impl<R: RSNN> Configurable for RSNNModel<R> {
    type Config = RSNNConfig<R>;
}

#[derive(Clone, Debug, Deserialize)]
pub struct RSNNConfig<R: RSNN> {
    pub n: usize,
    pub model: R::Config
}

impl<R: RSNN> ConfigSection for RSNNConfig<R> {
    fn name() -> String {
        "rsnn".to_string()
    }
}
