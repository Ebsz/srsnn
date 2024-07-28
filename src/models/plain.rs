//! The traditional model, where where the connections and synapses of the network are defined
//! explicitly.
//!

use crate::csa::{ConnectionSet, ValueSet, DynamicsSet};
use crate::csa::mask::Mask;
use crate::models::rsnn::{RSNN, RSNNConfig};

use utils::math;
use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};

use serde::Deserialize;
use ndarray::Array;

use std::sync::Arc;


const CONNECTIVITY_THRESHOLD: f32 = 1.0;
const W_WEIGHTING: f32 = 3.0;

#[derive(Clone, Debug)]
pub struct PlainModel;

impl RSNN for PlainModel {

    fn dynamics(_p: &ParameterSet, _config: &RSNNConfig<Self>) -> DynamicsSet {
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
        let w = b.mapv(|x| math::sigmoid(x) * W_WEIGHTING);

        ConnectionSet {
            m,
            v: vec![ValueSet(w)]
        }
    }

    fn params(config: &RSNNConfig<Self>) -> ParameterSet {
        let cm = Parameter::Matrix(Array::zeros((config.n, config.n)));
        let w = Parameter::Matrix(Array::zeros((config.n, config.n)));

        ParameterSet { set: vec![cm, w] }
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
