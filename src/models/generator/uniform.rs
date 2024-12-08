//! Uniform model. Serves as a baseline model by sweeping uniformly(ish)
//! over the rsnn space.

use crate::models::generator::{Generator, NetworkModel};
use crate::models::generator_model::ModelConfig;

use csa::{NetworkSet, ConnectionSet, ValueSet, NeuronSet, NeuronMask};

use model::neuron::izhikevich::IzhikevichParameters;

use utils::config::{ConfigSection, Configurable};
use utils::environment::Environment;
use utils::parameters::{Parameter, ParameterSet};

use rand::Rng;
use ndarray::array;

use serde::Deserialize;

use std::sync::Arc;


const MAX_W: f32 = 5.0;

#[derive(Clone, Debug)]
pub struct UniformModel;

impl Generator for UniformModel {
    fn get(p: &ParameterSet, config: &ModelConfig<Self>, env: &Environment) -> (NetworkSet, ConnectionSet) {
        let mut rng = rand::thread_rng();

        let p = rng.gen_range(0.0..1.0);
        let m = csa::mask::random(p);

        let d = uniform_dynamics();

        let w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let ns = NetworkSet {
            m,
            v: vec![w],
            d: vec![d]
        };

        let p = rng.gen_range(0.0..1.0);
        let input_mask = csa::mask::random(p);

        let input_w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let input_cs = ConnectionSet {
            m: input_mask,
            v: vec![input_w]
        };

        (ns, input_cs)
    }

    fn params(config: &ModelConfig<Self>, env: &Environment) -> ParameterSet {
        ParameterSet {
            set: vec![]
        }
    }
}

fn uniform_dynamics() -> NeuronSet {
    let r = IzhikevichParameters::RANGES;


    NeuronSet {f: Arc::new(
        move |_i| array![
            rand::thread_rng().gen_range(r[0].0..r[0].1),  // a
            rand::thread_rng().gen_range(r[1].0..r[1].1),     // b
            rand::thread_rng().gen_range(r[2].0..r[2].1),     // c
            rand::thread_rng().gen_range(r[3].0..r[3].1),     // d
            if rand::random::<f32>() > 0.5 { 1.0 } else { 0.0 }
        ]
    )}
}

#[derive(Clone, Debug, Deserialize)]
pub struct UniformModelConfig {

}

impl ConfigSection for UniformModelConfig {
    fn name() -> String {
        "uniform_model".to_string()
    }
}

impl Configurable for UniformModel {
    type Config = UniformModelConfig;
}
