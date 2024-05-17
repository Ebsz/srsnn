//! Stochastic models of RSNNs

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
pub struct StochasticGenome {
    pub n: usize,
}

impl Genome for StochasticGenome {
    type Config = StochasticGenomeConfig;

    fn new(env: &EvolutionEnvironment, config: &Self::Config) -> Self {
        let min_neurons = env.inputs + env.outputs;

        StochasticGenome {
            n: random::random_range((min_neurons, config.max_neurons))
        }
    }

    fn mutate(&mut self, _config: &Self::Config) {
        self.n += random::random_range((0,2));
    }

    fn crossover(&self, _other: &Self) -> Self {
        self.clone()
    }
}

impl EvolvableGenome for StochasticGenome {
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
pub struct StochasticGenomeConfig {
    pub max_neurons: usize,
}

impl ConfigSection for StochasticGenomeConfig {
    fn name() -> String {
        "stochastic_genome".to_string()
    }
}

//fn instantiate(matrix: Array2<f32>) -> Array2<f32> {
//
//
//}
//
//

mod base_model {
    use super::*;

    #[derive(Deserialize)]
    struct BaseModelConfig {
       max_neurons: usize,

       mut_p_probability: f32,
       mut_n_probability: f32,
    }

    impl ConfigSection for BaseModelConfig {

        fn name() -> String {
            "base_model".to_string()
        }
    }

    impl Default for BaseModelConfig {
        fn default() -> Self {
            BaseModelConfig {
                max_neurons: 128,
                mut_p_probability: 0.4,
                mut_n_probability: 0.6
            }
        }
    }

    struct BaseModelGenome {
        n: usize,
        p: f32
    }

    impl Genome for BaseModelGenome {
        type Config = BaseModelConfig;

        fn new(env: &EvolutionEnvironment, config: &BaseModelConfig) -> BaseModelGenome {
            BaseModelGenome {
                n: random::random_range((10, 20)),
                p: random::random_range((0.0, 1.0))
            }
        }

        // TODO: clean this up
        fn mutate(&mut self, config: &BaseModelConfig) {
            if random::random_range((0.0, 1.0)) <  config.mut_n_probability {

                let noise: f32 = random::random_sample(StandardNormal);

                self.n = (self.n as i32 + random::random_range::<i32>((-2, 3))) as usize; // Not inclusive so this is [-2, 2]
            }
            if random::random_range((0.0, 1.0)) <  config.mut_p_probability {
                self.p = self.p + random::random_sample::<f32, StandardNormal>(StandardNormal);


                // Restrict p to (0.0, 1.0)
                if self.p < 0.0 {
                    self.p = 0.0;
                } else if self.p > 1.0 {
                    self.p = 1.0;
                }
            }
        }

        fn crossover(&self, other: &BaseModelGenome) -> BaseModelGenome {
            BaseModelGenome {
                n: *random::random_choice::<usize>(&vec![self.n, other.n]),
                p: *random::random_choice::<f32>(&vec![self.p, other.p])
            }
        }
    }
}
