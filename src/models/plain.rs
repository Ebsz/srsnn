//! The traditional model, where where the connections and synapses of the network are defined
//! explicitly.
//!
// Vanilla Pure Default Direct Plain Plain

use crate::csa;
use crate::csa::{ConnectionSet, ValueSet, DynamicsSet};
use crate::csa::mask::Mask;

use crate::models::rsnn::RSNN;

use model::Model;

use utils::parameters::{Parameter, ParameterSet};
use utils::config::{Configurable, ConfigSection};

use serde::Deserialize;
use ndarray::Array;

use std::sync::Arc;


const CONNECTIVITY_THRESHOLD: f32 = 1.0;

#[derive(Clone, Debug)]
pub struct PlainModel;

impl RSNN for PlainModel {

    fn dynamics(config: &Self::Config, p: &ParameterSet) -> DynamicsSet {
        Self::default_dynamics()
    }

    fn connectivity(config: &Self::Config, p: &ParameterSet) -> Mask {
        let a = match &p.set[0] {
            Parameter::Matrix(x) => { x.clone() },
            _ => { panic!("invalid parameter set") }
        };

        Mask { f: Arc::new(
            move |i,j| if a[[i as usize, j as usize]] > CONNECTIVITY_THRESHOLD { true } else { false }
        )}
    }

    fn params(config: &Self::Config) -> ParameterSet {
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
    n: usize,
}


impl ConfigSection for PlainModelConfig {
    fn name() -> String {
        "immediate_model".to_string()
    }
}
