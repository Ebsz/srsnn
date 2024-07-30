use csa::{ConnectionSet, NeuronSet, NeuralSet};
use crate::models::rsnn::{RSNN, RSNNConfig};

use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};

use serde::Deserialize;


#[derive(Clone, Debug)]
pub struct ERModel;

impl RSNN for ERModel {
    fn get(params: &ParameterSet, config: &RSNNConfig<Self>) -> NeuralSet {
        let cs = Self::connectivity(params, config);

        let d = Self::dynamics(params, config);

        NeuralSet {
            m: cs.m,
            v: cs.v,
            d: vec![d]
        }
    }


    fn params(_config: &RSNNConfig<Self>) -> ParameterSet {
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
            v: vec![]
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
