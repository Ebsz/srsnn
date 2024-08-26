use crate::models::rsnn::{RSNN, RSNNConfig};

use csa::{ConnectionSet, NeuronSet, NetworkSet};
use csa::mask::Mask;

use utils::math;
use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};
use utils::environment::Environment;

use serde::Deserialize;


#[derive(Clone, Debug)]
pub struct ERModel;

impl RSNN for ERModel {
    fn params(_config: &RSNNConfig<Self>, _env: &Environment) -> ParameterSet {
        let p = Parameter::Scalar(0.0);
        let p_in = Parameter::Scalar(0.0);
        let p_out = Parameter::Scalar(0.0);

        ParameterSet {
            set: vec![p, p_in, p_out],
        }
    }

    fn get(params: &ParameterSet, config: &RSNNConfig<Self>) -> (NetworkSet, ConnectionSet, Mask) {
        let (p, p_in, p_out) = Self::parse_params(params);

        let cs = ConnectionSet {
            m: csa::mask::random(math::ml::sigmoid(p)),
            v: vec![Self::default_weights()]
        };

        let d = Self::dynamics(params, config);

        let ns = NetworkSet {
            m: cs.m,
            v: cs.v,
            d: vec![d]
        };

        let cs_in = ConnectionSet {
            m: csa::mask::random(math::ml::sigmoid(p_in)),
            v: vec![Self::default_weights()]
        };

        let out_mask = csa::mask::random(math::ml::sigmoid(p_out));

        (ns, cs_in, out_mask)
    }
}

impl ERModel {
    //fn connectivity(params: &ParameterSet, config: &RSNNConfig<Self>) -> NetworkSet {
    //}

    fn dynamics(_params: &ParameterSet, _config: &RSNNConfig<Self>) -> NeuronSet {
        Self::default_dynamics()
    }


    fn parse_params(params: &ParameterSet)  -> (f32, f32, f32) {
        assert!(params.set.len() == 3);

        let p: f32 = match &params.set[0] {
            Parameter::Scalar(x) => {*x},
            _ => { panic!("invalid parameter set") }
        };

        let p_in: f32 = match &params.set[1] {
            Parameter::Scalar(x) => {*x},
            _ => { panic!("invalid parameter set") }
        };
        let p_out: f32 = match &params.set[2] {
            Parameter::Scalar(x) => {*x},
            _ => { panic!("invalid parameter set") }
        };

        (p, p_in, p_out)
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
