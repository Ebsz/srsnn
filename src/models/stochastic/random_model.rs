//! The simplest form of stochastic model, parameterized by n, the number of neurons,
//! and p, the probability that two arbitrary neurons are connected.
//!
//! This model serves as a baseline by providing a uniform distribution
//! over the connectivity space

use crate::gen;
use crate::phenotype::{EvolvableGenome, Phenotype};

use model::network::SpikingNetwork;
use model::neuron::izhikevich::Izhikevich;
use model::synapse::BaseSynapse;
use model::synapse::representation::MatrixRepresentation;

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;

use utils::random;
use utils::config::ConfigSection;

use serde::Deserialize;

use ndarray::Array;

#[derive(Clone)]
pub struct RandomGenome {
    pub n: usize,
    pub p: f32,
    min_neurons: usize
}

impl Genome for RandomGenome {
    type Config = RandomGenomeConfig;

    fn new(env: &EvolutionEnvironment, config: &Self::Config) -> Self {
        let min_neurons = env.inputs + env.outputs;

        RandomGenome {
            n: random::random_range((min_neurons, config.max_neurons)),
            p: random::random_range((0.0, 1.0)),
            min_neurons
        }
    }

    fn mutate(&mut self, _config: &Self::Config) {
        let noise: i32 = random::random_range((-1, 2));

        if (self.n as i32 + noise) >= self.min_neurons as i32 {
            self.n = (self.n as i32 + noise) as usize;
        }
    }

    fn crossover(&self, _other: &Self) -> Self {
        self.clone()
    }
}

impl EvolvableGenome for RandomGenome {
    type Phenotype = Phenotype<SpikingNetwork<Izhikevich, BaseSynapse<MatrixRepresentation>>>;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype {
        //log::info!("n: {:?}", self.n);

        let model = Izhikevich::default(self.n);
        let synapse = gen::synapse_gen::from_probability(self.n, 0.1, Array::ones(self.n).mapv(|_: f32| false));

        let network = SpikingNetwork::new(model, synapse, env.inputs, env.outputs);
        Phenotype::new(network, env.clone())
    }
}

#[derive(Deserialize)]
pub struct RandomGenomeConfig {
    pub max_neurons: usize,
}

impl ConfigSection for RandomGenomeConfig {
    fn name() -> String {
        "stochastic_genome".to_string()
    }
}
