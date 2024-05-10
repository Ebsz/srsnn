use model::spikes::Spikes;
use model::neuron::NeuronModel;
use model::synapse::Synapse;
use model::record::{Record, RecordType, RecordDataType};

use tasks::{TaskInput, TaskOutput};
use tasks::task_runner::Runnable;

use utils::random;

use ndarray::{s, Array, Array1, Array2};
use rand::distributions::Uniform;


const SYNAPTIC_INPUT_SCALING: f32 = 18.0;
const RANDOM_FIRING_PROBABILITY: f32 = 0.01;

pub struct RunnableNetwork<N: NeuronModel, S: Synapse> {
    pub neurons: N,
    pub synapse: S,
    pub inputs: usize,
    pub outputs: usize,

    pub record: Record,
    pub recording: bool,

    synapse_spikes: Spikes,
    network_state: Spikes,

    // Optional energy input
    noise_input: Option<(f32, f32)>,
    random_firing: bool
}

impl<N: NeuronModel, S: Synapse> Runnable for RunnableNetwork<N, S> {
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
        self.neurons.reset();
    }
}

impl<N: NeuronModel, S: Synapse> RunnableNetwork<N, S> {
    pub fn new(neurons: N, synapse: S, inputs: usize, outputs: usize) -> RunnableNetwork<N, S> {
        let noise_input = None;
        let random_firing = false;

        let synapse_size = synapse.neuron_count();
        let network_size = neurons.len();

        RunnableNetwork {
            neurons,
            synapse,
            inputs,
            outputs,

            synapse_spikes: Spikes::new(synapse_size),
            network_state: Spikes::new(network_size),

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
        self.synapse_spikes.data.slice_mut(s![0..self.neurons.len()]).assign(&self.network_state.data);
    }

    fn get_network_input(&mut self, synaptic_input: &Array1<f32>) -> Array1<f32> {
        // Get input only for network neurons
        let mut network_input = synaptic_input.slice(s![0..self.neurons.len()]).to_owned();

        if let Some(n) = self.noise_input {
            network_input += &(random::random_vector(self.neurons.len(), Uniform::new(n.0, n.1)));
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
        let random_spikes: Array1<f32> = random::random_vector(self.neurons.len(), Uniform::new(0.0, 1.0))
            .mapv(|x| if x > (1.0 - RANDOM_FIRING_PROBABILITY) {1.0} else {0.0});

        self.network_state.data = (&self.network_state.as_float() + &random_spikes)
            .mapv(|x| if x != 0.0 { true } else { false });
    }

    pub fn enable_recording(&mut self) {
        self.recording = true;
    }
}
