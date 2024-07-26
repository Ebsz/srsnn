use crate::csa;
use crate::csa::DynamicsSet;
use crate::csa::mask::Mask;
use crate::models::rsnn::RSNN;

use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};

use serde::Deserialize;

use ndarray::array;

use std::sync::Arc;


#[derive(Clone, Debug)]
pub struct ERModel;

impl RSNN for ERModel {
    fn dynamics(config: &Self::Config, params: &ParameterSet) -> DynamicsSet {
        DynamicsSet { f: Arc::new(
            move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        )}
    }

    fn connectivity(config: &Self::Config, params: &ParameterSet) -> Mask {
        assert!(params.set.len() == 1);

        let p: f32 = match &params.set[0] {
            Parameter::Scalar(x) => {*x},
            _ => { panic!("invalid parameter set") }
        };

        csa::mask::random(p)
    }

    fn params(config: &Self::Config) -> ParameterSet {
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
