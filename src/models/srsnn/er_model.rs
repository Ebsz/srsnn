use crate::csa;
use crate::csa::{ConnectionSet, ValueSet, DynamicsSet};
use crate::models::rsnn::RSNN;

use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};

use serde::Deserialize;

use ndarray::Array;


#[derive(Clone, Debug)]
pub struct ERModel;

impl RSNN for ERModel {
    fn dynamics(_config: &Self::Config, _params: &ParameterSet) -> DynamicsSet {
        Self::default_dynamics()
    }

    fn connectivity(_config: &Self::Config, params: &ParameterSet) -> ConnectionSet {
        assert!(params.set.len() == 1);

        let p: f32 = match &params.set[0] {
            Parameter::Scalar(x) => {*x},
            _ => { panic!("invalid parameter set") }
        };

        ConnectionSet {
            m: csa::mask::random(p),
            v: vec![ValueSet(Array::ones((1, 1)))]
        }
    }

    fn params(_config: &Self::Config) -> ParameterSet {
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
