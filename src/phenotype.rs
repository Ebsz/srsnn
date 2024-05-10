use crate::network::RunnableNetwork;

use model::neuron::izhikevich::Izhikevich;
use model::synapse::{Synapse, BaseSynapse};
use model::synapse::representation::MatrixRepresentation;
use model::synapse::matrix_synapse::MatrixSynapse;
use model::record::{Record, RecordType, RecordDataType};

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;
use evolution::genome::matrix_genome::MatrixGenome;

use tasks::{TaskInput, TaskOutput};
use tasks::task_runner::Runnable;

use ndarray::{s, Array, Array1, Array2};


//TODO: Rename this to reflect that being phenotypeable is the focus
//TODO: Move to Evolution
pub trait EvolvableGenome: Genome + Sized {
    type Phenotype: Runnable;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype;
}

impl EvolvableGenome for MatrixGenome {
    type Phenotype = RunnableNetwork<Izhikevich, MatrixSynapse>;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype {
        // Number of neurons in the network
        //let network_size = g.network_size();
        let network_size = self.network_size();

        let synapse_matrix: Array2<f32> = self.connections.mapv(|(e, w)| if e {w} else {0.0});

        let neuron_types: Array1<f32> = Array::ones(synapse_matrix.shape()[0]);

        //let synapse_representation = MatrixRepresentation::new(synapse_matrix, neuron_types);
        //let synapse = BaseSynapse::new(synapse_representation);

        let synapse = MatrixSynapse::new(synapse_matrix, neuron_types);

        let model = Izhikevich::default(network_size);

        RunnableNetwork::new(model, synapse, env.inputs, env.outputs)
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
