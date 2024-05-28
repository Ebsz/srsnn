//! The base stochastic model, which is parameterized by a connection probability matrix
//! that defines the probability that two arbitrary neurons are connected.

use crate::models::Model;
use crate::models::stochastic::StochasticGenomeConfig;
use crate::gen::stochastic::sample_connection_probability_matrix;

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
    pub neurons: Array1<NeuronDescription<Izhikevich>>,
    pub inputs: usize,
    pub outputs: usize,
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
            neurons: Array::from_vec(nvec),
            inputs: env.inputs,
            outputs: env.outputs
        }
    }

    fn mutate(&mut self, _config: &Self::Config) {
        self.connection_probability.mutate_single_value(0.1, (0.0, 1.0));
    }

    fn crossover(&self, other: &Self) -> Self {
        // TODO: Implement proper crossover of neurons

        BaseStochasticGenome {
            connection_probability: self.connection_probability.point_crossover(&other.connection_probability),
            neurons: self.neurons.clone(),
            ..*self
        }
    }
}

impl BaseStochasticGenome {
    pub fn sample(&self) -> NetworkDescription<NeuronDescription<Izhikevich>> {
        let connection_mask = sample_connection_probability_matrix(&self.connection_probability.data);
        let weights: Array2<f32> = connection_mask.mapv(|v| v as f32);

        NetworkDescription::new(self.neurons.clone(), connection_mask, weights, self.inputs, self.outputs)
    }
}

impl Model for BaseStochasticGenome {
    fn develop(&self) -> NetworkDescription<NeuronDescription<Izhikevich>> {
        self.sample()
    }
}
