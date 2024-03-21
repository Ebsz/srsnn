use ndarray::{Array, Array1, Array2};

use crate::evolution::EvolutionEnvironment;


use crate::utils::{random_range, random_sample};
use ndarray_rand::rand_distr::StandardNormal;

const MAX_NEURONS: usize = 50;
const INITIAL_NEURON_COUNT_RANGE: (i32, i32) = (2, 5);
const INITIAL_CONNECTION_COUNT_RANGE: (i32, i32) = (3, 4);

/// A generic genome
/// The connections matrix is stored as [to, from]
#[derive(Debug)]
pub struct Genome {
    pub neurons: Vec<NeuronGene>,
    // Network neurons have id's [0, MAX_NEURONS]; ports have ID's [MAX_NEURONS, size]
    pub connections: Array2<(bool, f32)>,
    pub environment: EvolutionEnvironment
}

#[derive(Clone, Copy, Debug)]
pub enum NeuronType {
    Network,
    Output,
}

#[derive(Clone, Copy, Debug)]
pub struct NeuronGene {
    pub id: i32,
    pub ntype: NeuronType
}

impl Genome {
    pub fn new(env: &EvolutionEnvironment) -> Genome {
        let mut neurons: Vec<NeuronGene> = vec![];

        let size: usize = env.inputs + MAX_NEURONS;
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

        // Add a number of random connections between neurons
        let c = random_range(INITIAL_CONNECTION_COUNT_RANGE);

        let mut count = 0;
        loop {
            let i: usize = random_range((0, network_size)) as usize;
            let j: usize = random_range((0, network_size)) as usize;

            // TODO:  Doesn't check for existing connections
            // This is not as easy as checking connections[i,j].0, also have to
            // verify that c < (n^2-n), or we're not gonna have enough connections:)))
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

    }

    pub fn crossover(&self, other: Genome) {

    }

    fn mutate_connection(&mut self) {

    }

    fn mutate_add_neuron(&mut self) {

    }

    fn mutate_add_connection(&mut self) {

    }

}
