use crate::network::{Network, RunnableNetwork};

use model::spikes::Spikes;
use model::neuron::izhikevich::Izhikevich;
use model::synapse::{Synapse, BaseSynapse};
use model::synapse::representation::MatrixRepresentation;
use model::record::{Record, RecordType, RecordDataType};

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;
use evolution::genome::matrix_genome::MatrixGenome;

use tasks::{TaskInput, TaskOutput};
use tasks::task_runner::Runnable;

use ndarray::{s, Array, Array1, Array2};


pub struct Phenotype<N: Network> {
    network: N,
    env: EvolutionEnvironment,

    network_inputs: usize,
    network_outputs: usize,
}

impl<N: Network> Phenotype<N> {
    fn new(network: N, env: EvolutionEnvironment) -> Phenotype<N> {
        Phenotype {
            network,
            env: env.clone(),

            network_inputs: env.inputs,
            network_outputs: env.outputs,
        }
    }

    fn task_output_to_network_input(&self, output: TaskOutput) -> Spikes {
        // Ensure that the sensor input is boolean
        let task_data: Array1<bool> = output.data.mapv(|x| if x != 0.0 { true } else { false });

        let mut network_input = Spikes::new(self.network_inputs);
        network_input.data.assign(&task_data);

        network_input
    }

    fn network_state_to_task_input(&self, network_state: Spikes) -> Vec<TaskInput> {
        let mut task_inputs: Vec<TaskInput> = Vec::new();

        // Parse the firing state of output neurons to commands
        for i in 0..self.network_outputs {
            // TODO: unnecessary cast; make phenotype.outputs be usize
            // If the neurons assigned to be output fire, add the corresponding input
            if network_state.data[i as usize] {
                task_inputs.push(TaskInput { input_id: i as i32});
            }
        }

        task_inputs
    }

    fn environment(&self) -> &EvolutionEnvironment {
        &self.env
    }
}

impl<N: Network> Runnable for Phenotype<N> {
    fn step(&mut self, output: TaskOutput) -> Vec<TaskInput> {
        let network_input = self.task_output_to_network_input(output);

        let network_state = self.network.step(network_input); // network_state: len(N)

        self.network_state_to_task_input(network_state)
    }

    fn reset(&mut self) {
        //TODO: Add reset fn to network trait and implement
    }
}

pub trait EvolvableGenome: Genome + Sized {
    type Phenotype: Runnable;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype;
}

impl EvolvableGenome for MatrixGenome {
    type Phenotype = Phenotype<RunnableNetwork<Izhikevich, BaseSynapse<MatrixRepresentation>>>;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype {
        let network_size = self.network_size();

        let synapse_matrix: Array2<f32> = self.connections.mapv(|(e, w)| if e {w} else {0.0});

        let neuron_types: Array1<f32> = Array::ones(synapse_matrix.shape()[0]);

        //let synapse_representation = MatrixRepresentation::new(synapse_matrix, neuron_types);
        //let synapse = BaseSynapse::new(synapse_representation);

        let synapse_representation = MatrixRepresentation::new(synapse_matrix, neuron_types);
        let synapse = BaseSynapse::new(synapse_representation);

        let model = Izhikevich::default(network_size);

        let network = RunnableNetwork::new(model, synapse, env.inputs, env.outputs);

        Phenotype::new(network, env.clone())
    }
}


//trait PhenotypeBuilder {
//    fn from_genome(g: Genome) -> Network;
//    // Validate weight matrix:
//    // * Ensure it is the correct size (k + N)
//    // * Ensure no connections from output neurons, or to input neurons
//}

//pub fn from_graph_genome(g: &Genome) {
//    //TODO: this is very similiar to LinearSynapse - use?
//    let mut input_weights: HashMap<i32, Vec<(i32, f32)>> = HashMap::new(); 
//    let mut output_weights: HashMap<i32, Vec<(i32, f32)>> = HashMap::new(); 
//
//    let mut network_connections: Vec<ConnectionGene> = Vec::new();
//
//    let inputs = g.environment.inputs;
//    let outputs = g.environment.outputs;
//
//    let neuron_genes: Vec<NeuronGene> = g.genes.iter().filter_map(|g| match g {
//        GeneType::Neuron(n) => Some(*n),
//        _ => None
//    }).collect();
//
//    println!("{:?}", neuron_genes);
//
//    let connection_genes: Vec<ConnectionGene> = g.genes.iter().filter_map(|g| match g {
//        GeneType::Connection(c) => Some(*c),
//        _ => None
//    }).collect();
//
//    // Parse connections
//    for c in connection_genes {
//        // Skip disabled genes
//        if !c.enabled { 
//            continue;
//        }
//        
//        // Connections from input
//        if c.from < inputs { 
//            input_weights.entry(c.from).or_insert(Vec::new()).push((c.to, c.weight));
//        }
//
//        // Connections to output
//        if inputs <= c.to && c.to < (inputs+outputs) {
//            input_weights.entry(c.from).or_insert(Vec::new()).push((c.to, c.weight));
//        }
//
//        // Network connections
//        if c.to >= (inputs + outputs) && c.from >= (inputs + outputs) {
//            println!("{:?}", c);
//            network_connections.push(c);
//        }
//    }
//
//    let network_size = neuron_genes.len();
//    let mut weights: Array2<f32> = Array::zeros((network_size, network_size));
//
//    //for c in network_connections {
//    //    weights[
//
//    //}
//
//    println!("{:?}", weights);
//
//    panic!("disco");
//}

//fn parse_connections(connection_genes: Vec<ConnectionGene>) -> ({
//
//}


//mod test {
//
//}
