use ndarray::{s, Array, Array2};

use crate::EvolutionEnvironment;
use crate::config::GenomeConfig;

use utils::random::{random_range, random_sample, random_choice};
use ndarray_rand::rand_distr::StandardNormal;


/// An evolvable genome.
///
/// The connections matrix is stored as [to, from], and
/// has dim [N, (N + M)], where N is the number of neurons,
/// M is the number of inputs.
///
/// Network neurons have id's [0..MAX_NEURONS]; inputs have ID's [MAX_NEURONS..size].
///
#[derive(Clone, Debug)]
pub struct Genome {
    pub neurons: Vec<NeuronGene>,
    pub connections: Array2<(bool, f32)>,
}

impl Genome {
    pub fn new(env: &EvolutionEnvironment, config: &GenomeConfig) -> Genome {
        let mut neurons: Vec<NeuronGene> = vec![];

        let size: usize = config.max_neurons + env.inputs;
        let mut connections: Array2<(bool, f32)> = Array::zeros((size, size)).mapv(|_: i32| (false, 0.0));

        // Add output neurons
        for i in 0..env.outputs {
            neurons.push(NeuronGene {
                id: (i as u32),
                ntype: NeuronType::Output
            });
        }

        // Add network neurons
        for i in 0..random_range(config.initial_neuron_count_range) {
            neurons.push(NeuronGene {
                id: (i + env.outputs) as u32,
                ntype: NeuronType::Network
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

        Genome {
            neurons,
            connections,
        }
    }

    pub fn network_size(&self) -> usize {
        self.neurons.len()
    }

    /// Perform one of a set of different mutations on the genome
    pub fn mutate(&mut self, config: &GenomeConfig) {
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

    /// Return a new Genome that is a mix of two Genomes.
    ///
    /// The genome is created by iterating over each neuron and selecting
    /// its input connections from one of the genomes at random.
    ///
    ///
    /// let g = Genome::new();
    pub fn crossover(&self, other: &Genome) -> Genome{
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

        Genome {
            neurons,
            connections: connection_matrix,
        }
    }

    /// Selects a random connection and makes a small change
    fn mutate_connection(&mut self, _config: &GenomeConfig) {
        const MUTATION_STRENGTH: f32 = 0.5;

        let connection = self.get_random_connection();

        if let Some(c) = connection {
            let offset: f32 = random_sample(StandardNormal);
            let new_weight = self.connections[[c.0, c.1]].1 + offset * MUTATION_STRENGTH;
            self.connections[[c.0, c.1]] = (true, new_weight);
        }
    }

    /// Selects a random existing connection and flips its enable flag
    fn mutate_toggle_connection(&mut self, _config: &GenomeConfig) {
        let connection = self.get_random_connection();

        if let Some(c) = connection {
            self.connections[[c.0, c.1]].0 = !self.connections[[c.0, c.1]].0;
        }
    }

    /// Add new neuron to the genome
    fn mutate_add_neuron(&mut self, config: &GenomeConfig) {
        //NOTE: This only adds connections within the network,
        //      ie. not from input. Should this be allowed?

        if self.network_size() >= config.max_neurons {
            log::warn!("mutate_add_neuron on genome with max neurons");
            return;
        }

        let id = self.network_size() as u32;

        self.neurons.push( NeuronGene {
            id,
            ntype: NeuronType::Output
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
    fn mutate_add_connection(&mut self, _config: &GenomeConfig) {
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
    pub ntype: NeuronType
}

#[derive(Clone, Copy, Debug)]
pub enum NeuronType {
    Network,
    Output,
}


#[cfg(test)]
mod tests {
    use crate::evolution::genome::Genome;
    use crate::evolution::EvolutionEnvironment;

    #[test]
    fn test_crossover_correct_size() {
        let env = EvolutionEnvironment {
            inputs: 5,
            outputs: 2
        };

        let mut g1 = Genome::new(&env);
        let mut g2 = Genome::new(&env);
        let mut gc = g1.crossover(&g2);

        if g1.network_size() > g2.network_size() {
            assert_eq!(gc.network_size(), g1.network_size());
        } else {
            assert_eq!(gc.network_size(), g2.network_size());
        }

        assert_eq!(gc.connections.shape(), g1.connections.shape());
    }
}
