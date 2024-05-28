//! The main stochastic model from which all other stochastic models are derived.
//!
//! NOTE: The model is parameterized by N neurons, which encompass all non-input
//! neurons. The number of neurons in a sampled network is (N + env.inputs)
//!
//! TODO: Because type_connection_probabilities also contains the input type, mutation will also
//! mutate the entries of the matrix containing connections to input neurons. While this shouldn't
//! affect how the network runs, it still means that certain mutations will be "wasted" - which is
//! not ideal.
//!

use crate::models::Model;
use crate::gen::stochastic;

use model::network::SpikingNetwork;
use model::network::builder::NetworkBuilder;
use model::network::description::{NetworkDescription, NeuronDescription, NeuronRole};
use model::neuron::izhikevich::{Izhikevich, IzhikevichParameters};
use model::synapse::BaseSynapse;
use model::synapse::representation::MatrixRepresentation;

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;
use evolution::genome::representation::MatrixGene;

use utils::random;
use utils::math;
use utils::config::ConfigSection;

use std::collections::HashMap;

use serde::Deserialize;

use ndarray::{s, Array, Array1, Array2};


#[derive(Deserialize)]
pub struct MainModelConfig {
    pub n: usize,
    pub k: usize,
    pub type_inhibitory_probability: f32,

    pub n_mutations: usize,

    pub mutate_type_cpm_probability: f32,
    pub mutate_params_probability: f32,
    pub mutate_type_ratio_probability: f32,

    pub initial_probability_range: (f32, f32)
}

impl ConfigSection for MainModelConfig {
    fn name() -> String {
        "main_model".to_string()
    }
}

/// types contains the specified neuron types, excluding the input type.
///
/// type_connection_probabilities is a connection matrix of size (K + 1) containing the connection
/// probabilities between all types, including the input type
pub struct MainStochasticModel {
    pub n: usize, // NOTE: does not include inputs
    pub inputs: usize,
    pub outputs: usize,

    pub types: HashMap<u32, NeuronType>,
    pub type_connection_probabilities: MatrixGene,
}

impl Genome for MainStochasticModel {
    type Config = MainModelConfig;

    fn new(env: &EvolutionEnvironment, config: &MainModelConfig) -> MainStochasticModel {
        assert!(config.k <= config.n);

        let types = Self::init_neuron_types(config);

        let total_num_types = config.k + 1;

        MainStochasticModel {
            n: config.n,
            inputs: env.inputs,
            outputs: env.outputs,
            types,
            type_connection_probabilities: MatrixGene::init_random(total_num_types, config.initial_probability_range)
        }
    }

    fn mutate(&mut self, config: &MainModelConfig) {
        for _ in 0..config.n_mutations {
            if random::random_range((0.0, 1.0)) <  config.mutate_type_cpm_probability {
                self.type_connection_probabilities.mutate_single_value(0.1, (0.0, 1.0));
            }
            if random::random_range((0.0, 1.0)) <  config.mutate_params_probability {
                self.mutate_type_params(config);
            }
            if random::random_range((0.0, 1.0)) <  config.mutate_type_ratio_probability {
                self.mutate_type_ratio(config);
            }
        }
    }

    fn crossover(&self, other: &Self) -> Self {
        // TODO: implement for real

        MainStochasticModel {
            n: self.n,
            inputs: self.inputs,
            outputs: self.outputs,
            types: self.types.clone(),
            type_connection_probabilities: self.type_connection_probabilities.clone()
        }
    }
}

impl MainStochasticModel {

    fn mutate_type_ratio(&mut self, config: &MainModelConfig) {
        // TODO: Implement

    }
    fn mutate_type_params(&mut self, config: &MainModelConfig) {
        // TODO: Implement

    }


    fn init_neuron_types(config: &MainModelConfig) -> HashMap<u32, NeuronType> {
        // Create types
        let mut types = HashMap::new();

        for i in 0..config.k {
            types.insert(i as u32, NeuronType {
                prevalence: random::random_range((0.0, 1.0)),
                params: NeuronTypeParams::new(config),
            });
        }

        // Normalize type prevalances
        let p_sum: f32 = types.iter().map(|(k, t)| t.prevalence).sum();
        for (k, t) in &mut types {
            t.prevalence = t.prevalence / p_sum;
        }

        let prevalences: Vec<f32> = types.iter().map(|(k, t)| t.prevalence).collect();

        let total_prevalence = types.iter().fold(0.0f32, |acc, (k, t)| acc + t.prevalence);

        // We allow a tiny bit of fault because floating numbers and whatnot
        assert!((total_prevalence <= math::P_TOLERANCE), "total prevalence: {total_prevalence}");

        types
    }

    fn normalize_type_prevalences(types: HashMap<u32, NeuronType>) {
        // TODO: Implement
    }

    pub fn sample(&self) -> NetworkDescription<NeuronDescription<Izhikevich>> {
        // Number of neurons for each neuron type.
        let mut ndist = self.get_neuron_distribution();

        assert!(ndist.iter().sum::<usize>() == self.n + self.inputs);

        let neurons = self.generate_neurons(&ndist);

        let cpm = self.generate_connection_probability_matrix(&ndist);
        let mut theta: Array2<u32> = stochastic::sample_connection_probability_matrix(&cpm);

        // Remove incoming connections to input neurons
        theta.slice_mut(s![self.n..(self.n + self.inputs),..]).fill(0);

        let weights: Array2<f32> = theta.mapv(|v| v as f32);

        //println!("cpm size: {:?}", cpm.shape()[0]);
        //println!("neurons size: {:?}", neurons.shape()[0]);

        assert!(neurons.shape()[0] == self.n + self.inputs);

        NetworkDescription::new(neurons, theta, weights, self.inputs, self.outputs)
    }

    fn get_neuron_distribution(&self) -> Array1<usize> {
        let prevalences: Vec<f32> = self.types.iter().map(|(_, t)| t.prevalence).collect();

        let mut dist = math::distribute(self.n - self.outputs, &prevalences);

        // Add output neurons to the first type i.e. the output type
        dist[0] += self.outputs;

        dist.push(self.inputs);

        Array::from_vec(dist)
    }

    fn generate_neurons(&self, ndist: &Array1<usize>) -> Array1<NeuronDescription<Izhikevich>> {
        let mut neurons = Vec::new();

        let mut id = 0;

        // Add output and network neurons
        for (i, t) in self.types.iter() {

            // The first type is the output type.
            let role = if *i == 0 { NeuronRole::Output } else { NeuronRole::Network };

            for _ in 0..ndist[*i as usize] {
                neurons.push( NeuronDescription::new(
                        id,
                        Some(IzhikevichParameters { a: t.params.a, b: t.params.b, c: t.params.c, d: t.params.d, }),
                        t.params.inhibitory,
                        role));

                id += 1;
            }
        }

        // Add input neurons
        for _ in 0..self.inputs {
            neurons.push(NeuronDescription::new(id, None, false, NeuronRole::Input));
        }

        Array::from_vec(neurons)
    }


    fn generate_connection_probability_matrix(&self, ndist: &Array1<usize>) -> Array2<f32> {
        let n = self.n + self.inputs;
        let mut cpm = Array::ones((n,n));

        let p = &self.type_connection_probabilities.data;
        let n_types = ndist.shape()[0];

        let mut k: usize = 0;

        for i in 0..n_types {
            let mut l: usize = 0;

            let ic = ndist[i];

            for j in 0..n_types {
                let jc = ndist[j];

                cpm.slice_mut(s![k..(k+ic),l..(l+jc)]).fill(p[[i,j]]);

                l += jc;
            }
            k += ic;
        }

        cpm
    }
}

impl Model for MainStochasticModel {
    fn develop(&self) -> NetworkDescription<NeuronDescription<Izhikevich>> {
        self.sample()
    }
}

#[derive(Clone)]
pub struct NeuronType {
    prevalence: f32,
    params: NeuronTypeParams,
}

/// Parameters dictating the dynamics of a neuron type
#[derive(Clone)]
pub struct NeuronTypeParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,

    inhibitory: bool
}

impl NeuronTypeParams {
    const PARAM_RANGE: [(f32, f32); 4] = [(0.02, 0.1), (0.2, 0.25), (-65.0, -50.0), (2.0, 8.0)];

    /// Initialize a new set by randomly selecting params from within the allowed range
    fn new(config: &MainModelConfig) -> NeuronTypeParams {
        NeuronTypeParams {
            a: random::random_range((Self::PARAM_RANGE[0].0, Self::PARAM_RANGE[0].1)),
            b: random::random_range((Self::PARAM_RANGE[1].0, Self::PARAM_RANGE[1].1)),
            c: random::random_range((Self::PARAM_RANGE[2].0, Self::PARAM_RANGE[2].1)),
            d: random::random_range((Self::PARAM_RANGE[3].0, Self::PARAM_RANGE[3].1)),

            inhibitory: if random::random_range((0.0, 1.0)) >
                config.type_inhibitory_probability { false } else { true }
        }
    }
}