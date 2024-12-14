//! Family of models where connectivity is parameterized by a single value.

use crate::models::generator;
use crate::models::generator::{Generator, NetworkModel};
use crate::models::generator_model::ModelConfig;

use csa::{NetworkSet, ConnectionSet, ValueSet, NeuronSet, NeuronMask};

use utils::config::{ConfigSection, Configurable, EmptyConfig};
use utils::environment::Environment;
use utils::parameters::{Parameter, ParameterSet};

use rand::Rng;
use ndarray::array;

use serde::Deserialize;

use std::sync::Arc;


const MAX_W: f32 = 5.0;

#[derive(Clone, Debug)]
pub struct ER0Model;

impl Generator for ER0Model {
    fn get(ps: &ParameterSet, config: &ModelConfig<Self>, env: &Environment) -> (NetworkSet, ConnectionSet) {
        let mut rng = rand::thread_rng();

        let p: f32 = match &ps.set[0] {
            Parameter::Scalar(x) => {*x},
            _ => { panic!("invalid parameter set") }
        };

        let m = csa::mask::random(p);

        let d = generator::blk::dynamics::uniform();

        let w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let ns = NetworkSet {
            m,
            v: vec![w],
            d: vec![d]
        };

        let input_mask = csa::mask::random(p);

        let input_w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let input_cs = ConnectionSet {
            m: input_mask,
            v: vec![input_w]
        };

        (ns, input_cs)
    }

    fn params(config: &ModelConfig<Self>, env: &Environment) -> ParameterSet {
        let p = Parameter::Scalar(0.0);

        ParameterSet {
            set: vec![p]
        }
    }
}

impl Configurable for ER0Model {
    type Config = EmptyConfig;
}
