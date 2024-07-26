/// Generic model of a recurrent spiking neural network

use crate::csa::DynamicsSet;
use crate::csa::mask::Mask;

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
    fn dynamics(config: &Self::Config, p: &ParameterSet) -> DynamicsSet;
    fn connectivity(config: &Self::Config, p: &ParameterSet) -> Mask;

    fn params(config: &Self::Config) -> ParameterSet;

    fn default_dynamics() -> DynamicsSet {
        DynamicsSet { f: Arc::new(
            move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        )}
    }
}

pub struct RSNNModel<R: RSNN> {
    n: usize,
    conf: R::Config,

    params: ParameterSet,
    env: Environment,
}

impl<R: RSNN> Model for RSNNModel<R> {
    fn new(
        conf: &<RSNNModel<R> as Configurable>::Config,
        params: &ParameterSet,
        env: &Environment) -> Self {
        RSNNModel {
            n: conf.n,
            params: params.clone(),
            env: env.clone(),

            conf: conf.model.clone()
        }
    }

    fn develop(&self) -> DefaultRepresentation {
        let mask = R::connectivity(&self.conf, &self.params);
        let dynamics = R::dynamics(&self.conf, &self.params);

        let network_cm = mask.matrix(self.n);
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

        let network_w = network_cm.mapv(|v| v as f32);

        let mut input_cm: Array2<u32> = Array::zeros((self.n, self.env.inputs));

        for i in 0..self.env.inputs {
            input_cm[[i,i]] = 1;
        }

        let input_w: Array2<f32> = Array::ones((self.n, self.env.inputs));

        NetworkRepresentation::new(neurons.into(), network_cm, network_w, input_cm, input_w, self.env.clone())
    }

    fn params(config: &RSNNConfig<R>) -> ParameterSet {
        R::params(&config.model)
    }
}

impl<R: RSNN> Configurable for RSNNModel<R> {
    type Config = RSNNConfig<R>;
}

#[derive(Clone, Debug, Deserialize)]
pub struct RSNNConfig<R: RSNN> {
    n: usize,
    model: R::Config
}

impl<R: RSNN> ConfigSection for RSNNConfig<R> {
    fn name() -> String {
        "rsnn".to_string()
    }
}
