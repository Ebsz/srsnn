use ndarray::{s, Array, Array2};

use crate::genome::Genome;
use crate::EvolutionEnvironment;

use utils::random::{random_range, random_sample, random_choice};
use utils::config::ConfigSection;

use ndarray_rand::rand_distr::StandardNormal;

use serde::Deserialize;

/// The connections matrix is stored as [to, from], and
/// has dim [N, (N + M)], where N is the number of neurons,
/// M is the number of inputs.
///
/// Network neurons have id's [0..MAX_NEURONS]; inputs have ID's [MAX_NEURONS..size].
#[derive(Clone, Debug)]
pub struct MatrixGenome {
    pub neurons: Vec<NeuronGene>,
    pub connections: Array2<(bool, f32)>,
}

impl Genome for MatrixGenome {
    type Config = MatrixGenomeConfig;

    fn new(env: &EvolutionEnvironment, config: &MatrixGenomeConfig) -> MatrixGenome {
        let mut neurons: Vec<NeuronGene> = vec![];

        let size: usize = config.max_neurons + env.inputs;
        let mut connections: Array2<(bool, f32)> = Array::zeros((size, size)).mapv(|_: i32| (false, 0.0));

        // Add output neurons
        for i in 0..env.outputs {
            neurons.push(NeuronGene {
                id: (i as u32),
                ntype: NeuronType::Output,
                inhibitory: false
            });
        }

        // Add network neurons
        for i in 0..random_range(config.initial_neuron_count_range) {
            let inhibitory = if random_range((0.0, 1.0)) > config.inhibitory_probability { false } else { true };

            neurons.push(NeuronGene {
                id: (i + env.outputs) as u32,
                ntype: NeuronType::Network,
                inhibitory
            });
        }

        let network_size = neurons.len();

        // Connections from input neurons are randomly distributed among the network neurons
        for i in config.max_neurons..(config.max_neurons + env.inputs) {
            let j: usize = random_range((0, network_size)) as usize;

            connections[[j, i]] = (true, random_sample(StandardNormal));
        }

        let c = random_range(config.initial_connection_count_range);

        // Add a number of random connections between neurons
        //
        // NOTE: this doesn't check for existing connections
        //
        // This is not as easy as checking connections[i,j].0, also have to
        // verify that c < (n^2-n), or we're not gonna have enough connections:)))
        let mut count = 0;
        loop {
            let i: usize = random_range((0, network_size)) as usize;
            let j: usize = random_range((0, network_size)) as usize;

            if i== j {
                continue;
            }

            connections[[j, i]] = (true, random_sample(StandardNormal));

            count += 1;
            if count == c {
                break;
            }
        }

        assert!(connections.shape() == [(config.max_neurons + env.inputs), (config.max_neurons + env.inputs)]);

        MatrixGenome {
            neurons,
            connections,
        }
    }


    /// Perform one of a set of different mutations on the genome
    fn mutate(&mut self, config: &MatrixGenomeConfig) {
        for _ in 0..config.n_mutations {
            if random_range((0.0, 1.0)) <  config.mutate_connection_probability {
                self.mutate_connection(&config);
            }

            if random_range((0.0, 1.0)) <  config.mutate_toggle_connection_probability {
                self.mutate_toggle_connection(&config);
            }

            if random_range((0.0, 1.0)) <  config.mutate_add_connection_probability {
                self.mutate_add_connection(&config);
            }

            if random_range((0.0, 1.0)) < config.mutate_add_neuron_probability {
                self.mutate_add_neuron(&config);
            }
        }
    }

    /// The genome is created by iterating over each neuron and selecting
    /// its input connections from one of the genomes at random.
    ///
    /// let g = Genome::new();
    fn crossover(&self, other: &MatrixGenome) -> MatrixGenome{
        /*
         * Create connection matrix by randomly selecting rows
         * from each of the genomes
         */
        let msize = self.connections.shape()[0];

        let mut connection_matrix: Array2<(bool, f32)> =
            Array::zeros((msize, msize)).mapv(|_: i32| (false, 0.0 as f32));

        // TODO: This can result in the genome having no connections from
        //       the input, which could be a problem
        for i in 0..msize {
            if random_range((0.0, 1.0)) < 0.5 {
                connection_matrix.slice_mut(s![i,..])
                    .assign(&self.connections.slice(s![i,..]));
            } else {
                connection_matrix.slice_mut(s![i,..])
                    .assign(&other.connections.slice(s![i,..]));
            }
        }

        /*
         * Create neuron list by randomly selecting genes,
         * and finally taking the remaining genes from the
         * larger genome.
         */
        let mut neurons: Vec<NeuronGene> = Vec::new();

        // Ensure the lists are sorted by id before selection
        assert!(self.neurons.windows(2).all(|n| n[0].id < n[1].id));
        assert!(other.neurons.windows(2).all(|n| n[0].id < n[1].id));

        // Sort out larger/smaller list
        let mut t = vec![&self.neurons, &other.neurons];
        t.sort_by(|x, y| y.len().cmp(&x.len()));

        let larger: &Vec<NeuronGene> = t[0];
        let smaller: &Vec<NeuronGene> = t[1];

        assert!(larger.len() >= smaller.len());

        // Randomly take neurons from either of the genomes
        for i in 0..smaller.len() {
            if random_range((0.0, 1.0)) < 0.5 {
                neurons.push(self.neurons[i]);
            } else {
                neurons.push(other.neurons[i]);
            }
        }

        // Take the rest of neurons from the larger genome
        neurons.extend_from_slice(&larger[smaller.len()..]);

        assert!(neurons.len() == larger.len());

        MatrixGenome {
            neurons,
            connections: connection_matrix,
        }
    }
}

impl MatrixGenome {
    pub fn network_size(&self) -> usize {
        self.neurons.len()
    }

    /// Selects a random connection and makes a small change
    fn mutate_connection(&mut self, _config: &MatrixGenomeConfig) {
        const MUTATION_STRENGTH: f32 = 0.5;

        let connection = self.get_random_connection();

        if let Some(c) = connection {
            let offset: f32 = random_sample(StandardNormal);
            let new_weight = self.connections[[c.0, c.1]].1 + offset * MUTATION_STRENGTH;
            self.connections[[c.0, c.1]] = (true, new_weight);
        }
    }

    /// Selects a random existing connection and flips its enable flag
    fn mutate_toggle_connection(&mut self, _config: &MatrixGenomeConfig) {
        let connection = self.get_random_connection();

        if let Some(c) = connection {
            self.connections[[c.0, c.1]].0 = !self.connections[[c.0, c.1]].0;
        }
    }

    /// Add new neuron to the genome
    fn mutate_add_neuron(&mut self, config: &MatrixGenomeConfig) {
        //NOTE: This only adds connections within the network,
        //      ie. not from input. Should this be allowed?

        if self.network_size() >= config.max_neurons {
            log::warn!("mutate_add_neuron on genome with max neurons");
            return;
        }

        let id = self.network_size() as u32;
        let inhibitory = if random_range((0.0, 1.0)) > config.inhibitory_probability { false } else { true };

        self.neurons.push( NeuronGene {
            id,
            ntype: NeuronType::Network,
            inhibitory
        });


        let mut from: u32;
        let mut to: u32;

        let last_id = id - 1;
        loop {
            from = random_range((0, last_id as u32));
            to = random_range((0, last_id as u32));

            if from != id && to != id {
                break
            }
        }

        // Add a random incoming connection
        self.connections[[id as usize, from as usize]] = (true, random_sample(StandardNormal));

        // Add a random outgoing connection
        self.connections[[to as usize, id as usize]] = (true, random_sample(StandardNormal));
    }

    /// Mutate the genome by adding a new non-existing connection
    /// between two neurons
    fn mutate_add_connection(&mut self, _config: &MatrixGenomeConfig) {
        //NOTE: This only adds connections within the network,
        //      ie. not from input. Should this be allowed?

        let to: usize = random_range((0, self.network_size()));
        let mut from: usize;

        loop {
            from = random_range((0, self.network_size()));

            if from != to {
                break;
            }
        }

        self.connections[[to, from]] = (true, random_sample(StandardNormal));
    }

    /// Return the set of enabled connections (n1, n2)
    fn enabled_connections(&self) -> Vec<(usize, usize)> {
        self.connections.indexed_iter()
            .filter(|(_, x)| x.0)
            .map(|(i, _)| (i.0, i.1))
            .collect()
    }

    fn get_random_connection(&self) -> Option<(usize, usize)> {
        let enabled_connections = self.enabled_connections();

        if enabled_connections.len() == 0 {
            log::warn!("get_random_connection on genome with no enabled connections");
            return None;
        }

        Some(*random_choice(&enabled_connections))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NeuronGene {
    pub id: u32,
    pub ntype: NeuronType,
    pub inhibitory: bool
}

#[derive(Clone, Copy, Debug)]
pub enum NeuronType {
    Network,
    Output,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct MatrixGenomeConfig {
    pub max_neurons: usize,

    pub initial_neuron_count_range: (usize, usize),
    pub initial_connection_count_range: (usize, usize),

    pub n_mutations: usize,

    pub mutate_connection_probability: f32,
    pub mutate_toggle_connection_probability: f32,
    pub mutate_add_connection_probability: f32,
    pub mutate_add_neuron_probability: f32,

    pub inhibitory_probability: f32
}

impl ConfigSection for MatrixGenomeConfig {
    fn name() -> String {
        "matrix_genome".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EvolutionEnvironment;

    fn conf() -> MatrixGenomeConfig {
        MatrixGenomeConfig {
            max_neurons: 50,
            initial_neuron_count_range: (2, 5),
            initial_connection_count_range: (3, 4),
            n_mutations: 2,
            mutate_connection_probability: 0.8,
            mutate_toggle_connection_probability: 0.3,
            mutate_add_connection_probability: 0.03,
            mutate_add_neuron_probability: 0.02,
            inhibitory_probability: 0.0
        }
    }

    #[test]
    fn test_crossover_correct_size() {
        let env = EvolutionEnvironment {
            inputs: 5,
            outputs: 2,
        };

        let conf = conf();

        let g1 = MatrixGenome::new(&env, &conf);
        let g2 = MatrixGenome::new(&env, &conf);
        let gc = g1.crossover(&g2);

        if g1.network_size() > g2.network_size() {
            assert_eq!(gc.network_size(), g1.network_size());
        } else {
            assert_eq!(gc.network_size(), g2.network_size());
        }

        assert_eq!(gc.connections.shape(), g1.connections.shape());
    }
}
