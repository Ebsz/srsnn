//! The base model, which is parameterized by a connection probability matrix
//! that defines the probability that two arbitrary neurons are connected.

use crate::models::stochastic::StochasticGenomeConfig;
use crate::phenotype::{EvolvableGenome, Phenotype};

use model::network::SpikingNetwork;
use model::network::description::{NetworkDescription, NeuronDescription, NeuronRole};
use model::network::builder::NetworkBuilder;
use model::neuron::NeuronModel;
use model::neuron::izhikevich::{Izhikevich, IzhikevichParameters};
use model::synapse::BaseSynapse;
use model::synapse::representation::MatrixRepresentation;

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;
use evolution::genome::representation::MatrixGene;

use utils::random;

use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;


pub struct BaseStochasticGenome {
    pub connection_probability: MatrixGene,
    pub neurons: Array1<NeuronDescription<Izhikevich>>
}

impl Genome for BaseStochasticGenome {
    type Config = StochasticGenomeConfig;

    fn new(env: &EvolutionEnvironment, config: &Self::Config) -> Self {
        assert!(config.max_neurons >= (env.inputs + env.outputs));

        // Create neurons
        let mut nvec = Vec::new();

        for i in 0..config.max_neurons {
            let role: NeuronRole;
            let mut params = Some(IzhikevichParameters::default());

            if i < env.outputs {
                role = NeuronRole::Output;
            } else if i < (config.max_neurons - env.inputs) {
                role = NeuronRole::Network;
            } else {
                role = NeuronRole::Input;
                params = None;
            }

            nvec.push(NeuronDescription::new(i as u32, params, false, role));
        }

        let cpm = random::random_matrix((config.max_neurons, config.max_neurons), Uniform::new(0.0, 1.0));

        BaseStochasticGenome {
            connection_probability: MatrixGene { data: cpm },
            neurons: Array::from_vec(nvec)
        }
    }

    fn mutate(&mut self, _config: &Self::Config) {
        self.connection_probability.mutate_single_value(0.1, (0.0, 1.0));
    }

    fn crossover(&self, other: &Self) -> Self {
        // TODO: Implement proper crossover of neurons

        BaseStochasticGenome {
            connection_probability: self.connection_probability.point_crossover(&other.connection_probability),
            neurons: self.neurons.clone()
        }
    }
}

impl BaseStochasticGenome {
    pub fn sample(&self, env: &EvolutionEnvironment) -> NetworkDescription<Izhikevich> {
        let connection_mask = sample_connection_probability_matrix(&self.connection_probability.data);
        let weights: Array2<f32> = connection_mask.mapv(|v| v as f32);

        NetworkDescription::new(self.neurons.clone(), connection_mask, weights, env.inputs, env.outputs)
    }
}


impl EvolvableGenome for BaseStochasticGenome {
    type Phenotype = Phenotype<SpikingNetwork<Izhikevich, BaseSynapse<MatrixRepresentation>>>;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype {
        let description = self.sample(env);
        let network = NetworkBuilder::build(description);

        Phenotype::new(network, env.clone())
    }
}

/// Instantiate a connection probability matrix
pub fn sample_connection_probability_matrix(probability_matrix: &Array2<f32>) -> Array2<u32> {
    let n = probability_matrix.shape()[0];

    let samples: Array2<f32> = random::random_matrix((n,n), Uniform::new(0.0, 1.0));

    let mut connection_mask: Array2<u32> = Array::zeros((n,n));

    ndarray::Zip::from(&mut connection_mask)
        .and(probability_matrix)
        .and(&samples)
        .for_each(|c, p, s| *c = if s <= p {1} else {0} );

    connection_mask
}

#[test]
fn test_sampling_connection_probability_matrix() {
    let n = 10;

    let fully_connected = Array2::<u32>::ones((n,n));
    let no_connections = Array2::<u32>::zeros((n,n));

    let p1 = Array::ones((n,n));
    let c1 = sample_connection_probability_matrix(&p1);

    assert!(c1 == fully_connected);

    let p2 = Array::zeros((n,n));
    let c2 = sample_connection_probability_matrix(&p2);

    assert!(c2 == no_connections);
}
