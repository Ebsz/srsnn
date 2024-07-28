use crate::csa;
use crate::csa::{ConnectionSet, ValueSet, DynamicsSet};
use crate::models::rsnn::{RSNN, RSNNConfig};

use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};

use serde::Deserialize;

use ndarray::Array;


#[derive(Clone, Debug)]
pub struct ERModel;

impl RSNN for ERModel {
    fn dynamics(_params: &ParameterSet, _config: &RSNNConfig<Self>) -> DynamicsSet {
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

    fn params(_config: &RSNNConfig<Self>) -> ParameterSet {
        let p = Parameter::Scalar(0.0);

        ParameterSet {
            set: vec![p],
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
