//! The traditional model, where where the connections and synapses of the network are defined
//! explicitly.
//!

use csa::{ConnectionSet, ValueSet, NeuronSet, NeuralSet};
use csa::mask::Mask;
use crate::models::rsnn::{RSNN, RSNNConfig};

use utils::math;
use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};
use utils::environment::Environment;

use serde::Deserialize;
use ndarray::Array;

use std::sync::Arc;


const CONNECTIVITY_THRESHOLD: f32 = 1.0;
const W_WEIGHTING: f32 = 3.0;

#[derive(Clone, Debug)]
pub struct PlainModel;

impl RSNN for PlainModel {
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

    fn params(config: &RSNNConfig<Self>, _env: &Environment) -> ParameterSet {
        let cm = Parameter::Matrix(Array::zeros((config.n, config.n)));
        let w = Parameter::Matrix(Array::zeros((config.n, config.n)));

        ParameterSet { set: vec![cm, w] }
    }
}

impl PlainModel {
    fn dynamics(_p: &ParameterSet, _config: &RSNNConfig<Self>) -> NeuronSet {
        Self::default_dynamics()
    }

    fn connectivity(p: &ParameterSet, _config: &RSNNConfig<Self>) -> ConnectionSet {
        let a = match &p.set[0] {
            Parameter::Matrix(x) => { x.clone() },
            _ => { panic!("invalid parameter set") }
        };

        let b = match &p.set[1] {
            Parameter::Matrix(x) => { x },
            _ => { panic!("invalid parameter set") }
        };

        let m = Mask { f: Arc::new(
            move |i,j| if a[[i as usize, j as usize]] > CONNECTIVITY_THRESHOLD { true } else { false }
        )};

        // Create weights
        let w = b.mapv(|x| math::ml::sigmoid(x) * W_WEIGHTING);

        ConnectionSet {
            m,
            v: vec![ValueSet::from_value(w)]
        }
    }
}

impl Configurable for PlainModel {
    type Config = PlainModelConfig;
}


#[derive(Clone, Debug, Deserialize)]
pub struct PlainModelConfig {
}


impl ConfigSection for PlainModelConfig {
    fn name() -> String {
        "immediate_model".to_string()
    }
}
