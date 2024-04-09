use crate::model::neuron::izhikevich::Izhikevich;
use crate::model::synapse::matrix_synapse::MatrixSynapse;

use crate::evolution::EvolutionEnvironment;
use crate::evolution::genome::{Genome}; //, NeuronGene};

use ndarray::{Array, Array1, Array2};

// Idk?
//trait Phenotype {
//    fn from_genome(g: &Genome) -> Self;
//
//    fn inputs(&self) -> i32;
//    fn outputs(&self) -> i32;
//}

// Network implements phenotype


pub struct Phenotype {
    pub neurons: Izhikevich,
    pub synapse: MatrixSynapse,
    pub inputs: i32, //TODO: instead add env to phenotype?
    pub outputs: i32,
}

//impl Network<Izhikevich, MatrixSynapse> for Phenotype {
//    fn model(&mut self) -> Izhikevich {
//        self.neurons
//    }
//    fn synapses(&mut self) -> MatrixSynapse {
//        self.synapses
//    }
//}

impl Phenotype {
    pub fn from_genome(g: &Genome, env: &EvolutionEnvironment) -> Phenotype {
        // Number of neurons in the network
        let network_size = g.network_size();

        let synapse_matrix: Array2<f32> = g.connections.mapv(|(_, w)| w);

        let neuron_types: Array1<f32> = Array::ones(synapse_matrix.shape()[0]);
        let synapse = MatrixSynapse::new(synapse_matrix, neuron_types);
        let model = Izhikevich::default(network_size);

        Phenotype {
            neurons: model,
            synapse,
            inputs: env.inputs as i32,
            outputs: env.outputs as i32
        }
    }

    pub fn reset(&mut self) {
        self.neurons.initialize();
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
