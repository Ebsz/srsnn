use ndarray::{s, Array, Array1, Array2};

use crate::evolution::EvolutionEnvironment;

use crate::utils::{random_range, random_sample, random_choice};
use ndarray_rand::rand_distr::StandardNormal;

const MAX_NEURONS: usize = 13;

const INITIAL_NEURON_COUNT_RANGE: (i32, i32) = (2, 5);
const INITIAL_CONNECTION_COUNT_RANGE: (i32, i32) = (3, 4);

/*
 * Probabilities for each type of mutation
 */
const MUTATE_CONNECTION_PROB: f32 = 0.10;
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
#[derive(Debug)]
pub struct Genome {
    pub neurons: Vec<NeuronGene>,
    pub connections: Array2<(bool, f32)>,
    pub environment: EvolutionEnvironment
}

impl Genome {
    pub fn new(env: &EvolutionEnvironment) -> Genome {
        let mut neurons: Vec<NeuronGene> = vec![];

        let size: usize = MAX_NEURONS + env.inputs;

        // TODO: Change to be of size (MAX_NEURONS, MAX_NEURONS + inputs)
        let mut connections: Array2<(bool, f32)> = Array::zeros((size, size)).mapv(|_: i32| (false, 0.0));

        // Add output neurons
        for i in 0..env.outputs {
            neurons.push(NeuronGene {
                id: (i as i32),
                ntype: NeuronType::Output
            });
        }

        // Add network neurons
        for i in 0..random_range(INITIAL_NEURON_COUNT_RANGE) {
            neurons.push(NeuronGene {
                id: i + (env.outputs as i32),
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
        // verify that c < (n^2-n), or we're not gonna have enough connections:)))u
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
            environment: env.clone(),
        }
    }

    pub fn network_size(&self) -> usize {
        self.neurons.len()
    }

    pub fn mutate(&mut self) {
        if random_range((0.0, 1.0)) <  MUTATE_CONNECTION_PROB {
            self.mutate_connection();
        }

        if random_range((0.0, 1.0)) <  MUTATE_ADD_CONNECTION_PROB {
            self.mutate_add_connection();
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
            environment: self.environment.clone()
        }
    }

    fn mutate_connection(&mut self) {

    }

    fn mutate_add_neuron(&mut self) {

    }

    fn mutate_add_connection(&mut self) {

    }

#[derive(Clone, Copy, Debug)]
pub struct NeuronGene {
    pub id: i32,
    pub ntype: NeuronType
}

#[derive(Clone, Copy, Debug)]
pub enum NeuronType {
    Network,
    Output,
}
