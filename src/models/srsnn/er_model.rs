use crate::models::rsnn::{RSNN, RSNNConfig};

use csa::{ConnectionSet, NeuronSet, NeuralSet};
use csa::mask::Mask;

use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};
use utils::environment::Environment;

use serde::Deserialize;


#[derive(Clone, Debug)]
pub struct ERModel;

impl RSNN for ERModel {
    fn get(params: &ParameterSet, config: &RSNNConfig<Self>) -> (NeuralSet, ConnectionSet, Mask) {
        let cs = Self::connectivity(params, config);

        let d = Self::dynamics(params, config);

        let ns = NeuralSet {
            m: cs.m,
            v: cs.v,
            d: vec![d]
        };

        (ns, Self::default_input(config.n), Self::default_output())
    }

    fn params(_config: &RSNNConfig<Self>, _env: &Environment) -> ParameterSet {
        let p = Parameter::Scalar(0.0);

        ParameterSet {
            set: vec![p],
        }
    }
}

impl ERModel {
    fn dynamics(_params: &ParameterSet, _config: &RSNNConfig<Self>) -> NeuronSet {
        Self::default_dynamics()
    }

    fn connectivity(params: &ParameterSet, config: &RSNNConfig<Self>) -> ConnectionSet {
        assert!(params.set.len() == 1);

        let p: f32 = match &params.set[0] {
            Parameter::Scalar(x) => {*x},
            _ => { panic!("invalid parameter set") }
        };



        ConnectionSet {
            m: csa::mask::random(p),
            v: vec![Self::default_weights()]
        }
    }
}

impl Configurable for ERModel {
    type Config = ERConfig;
}

#[derive(Clone, Debug, Deserialize)]
pub struct ERConfig {
}

impl ConfigSection for ERConfig {
    fn name() -> String {
        "er_model".to_string()
    }
}
