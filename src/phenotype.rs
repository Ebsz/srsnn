use model::neuron::NeuronModel;
use model::neuron::izhikevich::Izhikevich;
use model::synapse::Synapse;
use model::synapse::matrix_synapse::MatrixSynapse;
use model::spikes::Spikes;
use model::record::{Record, RecordType, RecordDataType};

use evolution::EvolutionEnvironment;

use evolution::genome::Genome;
use evolution::genome::matrix_genome::MatrixGenome;

use tasks::{TaskInput, TaskOutput};
use tasks::task_runner::Runnable;

use utils::random;

use ndarray::{s, Array, Array1, Array2};
use rand::distributions::Uniform;


const SYNAPTIC_INPUT_SCALING: f32 = 18.0;
const RANDOM_FIRING_PROBABILITY: f32 = 0.01;

//pub trait ToPhenotype {
//    type Phenotype: Runnable;
//
//    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype;
//}

//TODO: Rename this to reflect that being phenotypeable is the focus
pub trait EvolvableGenome: Genome + Sized {
    type Phenotype: Runnable;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype;
}

//impl<Izhikevich, MatrixSynapse> Network<Izhikevich


pub struct MatrixPhenotype {
    pub neurons: Izhikevich,
    pub synapse: MatrixSynapse,
    pub inputs: usize,
    pub outputs: usize,

    pub record: Record,
    pub recording: bool,

    network_size: usize,
    synapse_spikes: Spikes,
    network_state: Spikes,

    // Optional energy input
    noise_input: Option<(f32, f32)>,
    random_firing: bool
}

impl Runnable for MatrixPhenotype {
    fn step(&mut self, task_output: TaskOutput) -> Vec<TaskInput> {
        self.get_synapse_spikes(task_output.data);
        let synaptic_input = self.synapse.step(&self.synapse_spikes) * SYNAPTIC_INPUT_SCALING;

        let network_input = self.get_network_input(&synaptic_input);
        self.network_state = self.neurons.step(network_input);

        if self.random_firing {
            self.add_random_output_firing();
        }

        if self.recording {
            self.record.log(RecordType::Potentials, RecordDataType::Potentials(self.neurons.potentials()));
            self.record.log(RecordType::Spikes, RecordDataType::Spikes(self.network_state.as_float()));
        }

        self.task_input()
    }

    fn reset(&mut self) {
        self.neurons.initialize();
    }
}

impl EvolvableGenome for MatrixGenome {
    type Phenotype = MatrixPhenotype;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> MatrixPhenotype {
        // Number of neurons in the network
        //let network_size = g.network_size();
        let network_size = self.network_size();

        let synapse_matrix: Array2<f32> = self.connections.mapv(|(e, w)| if e {w} else {0.0});

        let neuron_types: Array1<f32> = Array::ones(synapse_matrix.shape()[0]);
        let synapse = MatrixSynapse::new(synapse_matrix, neuron_types);
        let model = Izhikevich::default(network_size);

        MatrixPhenotype::new(model, synapse, env.inputs, env.outputs)
    }
}


impl MatrixPhenotype {
    pub fn new(neurons: Izhikevich, synapse: MatrixSynapse, inputs: usize, outputs: usize) -> MatrixPhenotype {
        let noise_input = None;
        let random_firing = false;

        let network_size = neurons.size();
        let synapse_size = synapse.neuron_count();

        MatrixPhenotype {
            neurons,
            synapse,
            inputs,
            outputs,

            synapse_spikes: Spikes::new(synapse_size),
            network_state: Spikes::new(network_size),
            network_size,

            noise_input,
            random_firing,

            record: Record::new(),
            recording: false
        }
    }

    fn get_synapse_spikes(&mut self, sensors: Array1<f32>) {
        // Read sensor data
        // NOTE: Ensures that the sensor input is boolean. This should be done elsewhere
        let sensor_data: Array1<bool> = sensors.mapv(|x| if x != 0.0 { true } else { false });

        // Assign sensor input and previous state to the the spike input for synapses
        self.synapse_spikes.data.slice_mut(s![(-(self.inputs as i32))..]).assign(&sensor_data);
        self.synapse_spikes.data.slice_mut(s![0..self.network_size]).assign(&self.network_state.data);
    }

    fn get_network_input(&mut self, synaptic_input: &Array1<f32>) -> Array1<f32> {
        // Get input only for network neurons
        let mut network_input = synaptic_input.slice(s![0..self.network_size]).to_owned();

        if let Some(n) = self.noise_input {
            network_input += &(random::random_vector(self.network_size, Uniform::new(n.0, n.1)));
        }

        network_input
    }

    fn task_input(&mut self) -> Vec<TaskInput> {
        let mut task_inputs: Vec<TaskInput> = Vec::new();

        // Parse the firing state of output neurons to commands
        for i in 0..self.outputs {
            // TODO: unnecessary cast; make phenotype.outputs be usize
            // If the neurons assigned to be output fire, add the corresponding input
            if self.network_state.data[i as usize] {
                task_inputs.push(TaskInput { input_id: i as i32});
            }
        }

        task_inputs
    }

    fn add_random_output_firing(&mut self) {
        let random_spikes: Array1<f32> = random::random_vector(self.network_size, Uniform::new(0.0, 1.0))
            .mapv(|x| if x > (1.0 - RANDOM_FIRING_PROBABILITY) {1.0} else {0.0});

        self.network_state.data = (&self.network_state.as_float() + &random_spikes)
            .mapv(|x| if x != 0.0 { true } else { false });
    }

    pub fn enable_recording(&mut self) {
        self.recording = true;
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
