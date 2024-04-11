use ndarray::{s, Array, Array2};

use crate::EvolutionEnvironment;

use utils::random::{random_range, random_sample, random_choice};
use ndarray_rand::rand_distr::StandardNormal;

const MAX_NEURONS: usize = 13;

const INITIAL_NEURON_COUNT_RANGE: (usize, usize) = (2, 5);
const INITIAL_CONNECTION_COUNT_RANGE: (usize, usize) = (3, 4);

/*
 * Probabilities for each type of mutation
 */
const MUTATE_CONNECTION_PROB: f32 = 0.80;
const MUTATE_TOGGLE_CONNECTION_PROB: f32 = 0.30;
const MUTATE_ADD_CONNECTION_PROB: f32 = 0.03;
const MUTATE_ADD_NEURON_PROB: f32 = 0.02;


///
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
    pub fn new(env: &EvolutionEnvironment) -> Genome {
        let mut neurons: Vec<NeuronGene> = vec![];

        let size: usize = MAX_NEURONS + env.inputs;
        let mut connections: Array2<(bool, f32)> = Array::zeros((size, size)).mapv(|_: i32| (false, 0.0));

        // Add output neurons
        for i in 0..env.outputs {
            neurons.push(NeuronGene {
                id: (i as u32),
                ntype: NeuronType::Output
            });
        }

        // Add network neurons
        for i in 0..random_range(INITIAL_NEURON_COUNT_RANGE) {
            neurons.push(NeuronGene {
                id: (i + env.outputs) as u32,
                ntype: NeuronType::Network
            });
        }

        let network_size = neurons.len();

        // Connections from input neurons are randomly distributed among the network neurons
        for i in MAX_NEURONS..(MAX_NEURONS + env.inputs) {
            let j: usize = random_range((0, network_size)) as usize;

            connections[[j, i]] = (true, random_sample(StandardNormal));
        }

        let c = random_range(INITIAL_CONNECTION_COUNT_RANGE);

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

        assert!(connections.shape() == [(MAX_NEURONS + env.inputs), (MAX_NEURONS + env.inputs)]);

        Genome {
            neurons,
            connections,
        }
    }

    pub fn network_size(&self) -> usize {
        self.neurons.len()
    }

    /// Perform one of a set of different mutations on the genome
    pub fn mutate(&mut self) {
        if random_range((0.0, 1.0)) <  MUTATE_CONNECTION_PROB {
            self.mutate_connection();
        }

        if random_range((0.0, 1.0)) <  MUTATE_TOGGLE_CONNECTION_PROB {
            self.mutate_toggle_connection();
        }

        if random_range((0.0, 1.0)) <  MUTATE_ADD_CONNECTION_PROB {
            self.mutate_add_connection();
        }

        if random_range((0.0, 1.0)) < MUTATE_ADD_NEURON_PROB {
            self.mutate_add_neuron();
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
    fn mutate_connection(&mut self) {
        const MUTATION_STRENGTH: f32 = 0.5;

        let connection = self.get_random_connection();

        if let Some(c) = connection {
            let offset: f32 = random_sample(StandardNormal);
            let new_weight = self.connections[[c.0, c.1]].1 + offset * MUTATION_STRENGTH;
            self.connections[[c.0, c.1]] = (true, new_weight);
        }
    }

    /// Selects a random existing connection and flips its enable flag
    fn mutate_toggle_connection(&mut self) {
        let connection = self.get_random_connection();

        if let Some(c) = connection {
            self.connections[[c.0, c.1]].0 = !self.connections[[c.0, c.1]].0;
        }
    }

    /// Mutate the genome by adding a new neuron
    fn mutate_add_neuron(&mut self) {
        //NOTE: This only adds connections within the network,
        //      ie. not from input. Should this be allowed?
        let id = self.neurons.len() as u32;

        self.neurons.push( NeuronGene {
            id,
            ntype: NeuronType::Output
        });

        let mut from: usize;
        let mut to: usize;

        let ns = self.network_size();

        loop {
            from = random_range((0, ns));
            to = random_range((0, ns));

            if from != id as usize && to != id as usize {
                break
            }
        }

        // Add a random incoming connection
        self.connections[[id as usize, from]] = (true, random_sample(StandardNormal));

        // Add a random outgoing connection
        self.connections[[to, id as usize]] = (true, random_sample(StandardNormal));
    }

    /// Mutate the genome by adding a new non-existing connection
    /// between two neurons
    fn mutate_add_connection(&mut self) {
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
