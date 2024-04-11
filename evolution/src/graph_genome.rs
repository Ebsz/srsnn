use ndarray::{Array, Array1, Array2};

use crate::evolution::EvolutionEnvironment;


use crate::utils::{random_range, random_sample};
use ndarray_rand::rand_distr::StandardNormal;

const MAX_GENOME_SIZE: usize = 20;

// The 
const PORT_ID_OFFSET: i32 = 255;

// TODO: implement this later
//pub trait Genome {
//    fn mutate(&mut self);
//
//}


#[derive(Debug)]
pub struct Genome {
    pub genes: Vec<GeneType>,
    pub environment: EvolutionEnvironment
}


#[derive(Clone, Copy, Debug)]
pub struct ConnectionGene {
    pub from: i32,
    pub to: i32,
    pub weight: f32,
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct NeuronGene {
    pub id: i32
}


#[derive(Debug)]
pub enum GeneType {
    Neuron(NeuronGene),
    Connection(ConnectionGene)
}

impl Genome {
    pub fn new(env: &EvolutionEnvironment) -> Genome {
        const INITIAL_NEURON_COUNT_RANGE: (i32, i32) = (2, 10);
        const INITIAL_CONNECTION_COUNT_RANGE: (i32, i32) = (2, 5);

        let mut genes: Vec<GeneType> = vec![];

        let n = random_range(INITIAL_NEURON_COUNT_RANGE);

        // NOTE: Input/output neurons have IDs in the range [0, inputs], outputs: [inputs, [inputs+outputs];
        // network neurons have id's > (inputs+outputs)
        // TODO: Idea: have a different start-id for ports (e.g. 255)?
        let k = env.inputs + env.outputs;
        for i in 0..n {
            genes.push(GeneType::Neuron(NeuronGene {
                id: k+(i as i32)

            }));
        }

        // Connections from input neurons are randomly distributed among the network neurons
        for i in 0..env.inputs {
            let j = random_range((k, k+n));

            genes.push(GeneType::Connection(ConnectionGene {
                from: i,
                to: j,
                weight: random_sample(StandardNormal),
                enabled: true
            }));
        }

        for i in env.inputs..(env.inputs+env.outputs) {
            let j = random_range((k, k+n));

            genes.push(GeneType::Connection(ConnectionGene {
                from: j,
                to: i,
                weight: random_sample(StandardNormal),
                enabled: true
            }));
        }

        // TODO: calling with an integer range seems to not include 
        //       the last number of the range; investigate.
        let c = random_range(INITIAL_CONNECTION_COUNT_RANGE);

        let mut count = 0;
        // TODO:  Doesn't check for existing connections; fix
        loop {
            let i = random_range((k, k+n));
            let j = random_range((k, k+n));

            if i== j {
                continue;
            }

            genes.push(GeneType::Connection(ConnectionGene {
                from: i,
                to: j,
                weight: random_sample(StandardNormal),
                enabled: true
            }));
            count += 1;
            if count == c {
                break;
            }
        }

        Genome {
            genes,
            environment: env.clone(),
        }
    }

    pub fn mutate(&mut self) { }


    fn mutate_connection(&mut self) { }

    fn mutate_add_neuron(&mut self) { }

    fn mutate_add_connection(&mut self) { }

    pub fn crossover(&self, other: Genome) { }
}
