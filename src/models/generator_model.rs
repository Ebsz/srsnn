use crate::models::generator::Generator;



use model::Model;
use model::network::representation::{DefaultRepresentation, NetworkRepresentation, NeuronDescription};
use model::neuron::izhikevich::IzhikevichParameters;

use utils::parameters::ParameterSet;
use utils::environment::Environment;
use utils::config::{Configurable, ConfigSection};

use serde::Deserialize;

use std::fmt::Debug;


/// A fully defined network generator that can be sampled repeatedly
pub struct GeneratorModel<G: Generator> {
    n: usize,
    conf: ModelConfig<G>,

    params: ParameterSet,
    env: Environment,
}

impl<G: Generator> Model for GeneratorModel<G> {
    fn new(
        conf: &ModelConfig<G>,
        params: &ParameterSet,
        env: &Environment) -> Self {
        GeneratorModel {
            n: conf.n,
            params: params.clone(),
            env: env.clone(),

            conf: conf.clone()
        }
    }

    fn develop(&self) -> DefaultRepresentation {
        let n = self.n + self.env.outputs;

        let (neural_set, input_cs) = G::get(&self.params, &self.conf, &self.env);

        let mask = neural_set.m;

        // Self connections are always removed
        let network_cm = (mask - csa::mask::one_to_one()).matrix(n);

        let dynamics = &neural_set.d[0];
        let d = dynamics.vec(n);

        let mut neurons = Vec::new();
        for i in 0..n {
            let inhibitory = if d[i][4] == 1.0 { true } else { false };
            neurons.push(NeuronDescription::new(
                    i as u32,
                    IzhikevichParameters {
                        a: d[i][0],
                        b: d[i][1],
                        c: d[i][2],
                        d: d[i][3],
                    },
                    inhibitory,
            ));
        }

        let network_w = neural_set.v[0].matrix(n);

        let input_cm = input_cs.m.r_matrix(self.n, self.env.inputs);
        let input_w = input_cs.v[0].r_matrix(self.n, self.env.inputs);

        if input_cm.iter().all(|c| *c == 0) {
            log::trace!("No connections from input");
        }

        NetworkRepresentation::new(neurons.into(),
            network_cm,
            network_w,
            input_cm,
            input_w,
            self.env.clone())
    }

    fn params(config: &ModelConfig<G>, env: &Environment) -> ParameterSet {
        G::params(config, env)
    }
}

impl<G: Generator> Configurable for GeneratorModel<G> {
    type Config = ModelConfig<G>;
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig<G: Generator> {
    pub n: usize,
    pub model: G::Config
}

impl<G: Generator> ConfigSection for ModelConfig<G> {
    fn name() -> String {
        "generator".to_string()
    }
}
