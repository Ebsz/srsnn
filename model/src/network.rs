use crate::neuron::NeuronModel;
use crate::spikes::Spikes;
use crate::synapse::Synapse;
use crate::record::{Record, RecordType, RecordDataType};

use utils::random;

use ndarray::{s, Array1, Array2};
use rand::distributions::Uniform;

use std::time::Instant;

// Output of a neuron is multiplied by this; determines the impact of a single spike.
const SYNAPTIC_INPUT_SCALING: f32 = 18.0; // 15.0;
const RANDOM_FIRING_PROBABILITY: f32 = 0.01;


pub trait Network {
    fn step(&mut self, input: Spikes) -> Spikes;
}


/// A network contains neurons of a specific model N that are
/// connected via a specific type of synapse S
pub trait RunnableNetwork<N: NeuronModel, S: Synapse> {
    fn step(&mut self, input: Array1<f32>, state: Spikes) -> Spikes {
        let synaptic_input = self.synapse().step(&state) * SYNAPTIC_INPUT_SCALING;

        let step_input = synaptic_input + &input;

        self.model().step(step_input)
    }

    /// Run the network on pre-defined input
    fn run(&mut self, steps: usize, input: &Array2<f32>) -> Record {
        log::trace!("Running network..");
        let n_neurons = self.model().potentials().shape()[0];
        assert!(input.shape() == [steps, n_neurons]);

        let mut firing_state = Spikes::new(n_neurons);
        let mut record = Record::new();

        let start_time = Instant::now();
        for i in 0..steps {
            let external_input = input.slice(s![i, ..]).to_owned();

            firing_state = self.step(external_input, firing_state);

            record.log(RecordType::Potentials, RecordDataType::Potentials(self.model().potentials()));
            record.log(RecordType::Spikes, RecordDataType::Spikes(firing_state.as_float()));
        }

        log::trace!("Simulated {} neurons for {} steps in {}s", n_neurons, steps, (start_time.elapsed().as_secs_f32()));

        record
    }

    fn model(&mut self) -> &mut N; // Rename to neuron
    fn synapse(&mut self) -> &mut S;
}

pub struct SpikingNetwork<N: NeuronModel, S: Synapse> {
    pub neurons: N,
    pub synapse: S,
    pub inputs: usize,
    pub outputs: usize,

    pub record: Record,
    pub recording: bool,

    network_state: Spikes,

    // Optional energy input
    noise_input: Option<(f32, f32)>,
    random_firing: bool
}

impl<N: NeuronModel, S: Synapse> Network for SpikingNetwork<N, S> {
    fn step(&mut self, input: Spikes) -> Spikes {
        assert!(input.len() == self.inputs);

        let full_network_state = self.get_full_network_state(input);
        assert!(full_network_state.len() == self.synapse.neuron_count());

        let synaptic_input = self.synapse.step(&full_network_state) * SYNAPTIC_INPUT_SCALING;

        let network_input = self.get_network_input(&synaptic_input);
        self.network_state = self.neurons.step(network_input);

        if self.random_firing {
            self.add_random_output_firing();
        }

        if self.recording {
            self.record.log(RecordType::Potentials, RecordDataType::Potentials(self.neurons.potentials()));
            self.record.log(RecordType::Spikes, RecordDataType::Spikes(self.network_state.as_float()));
        }

        self.network_state.clone()
    }
}

impl<N: NeuronModel, S: Synapse> SpikingNetwork<N, S> {
    pub fn new(neurons: N, synapse: S, inputs: usize, outputs: usize) -> SpikingNetwork<N, S> {
        let noise_input = None;
        let random_firing = false;

        let network_size = neurons.len();

        SpikingNetwork {
            neurons,
            synapse,
            inputs,
            outputs,

            network_state: Spikes::new(network_size),

            noise_input,
            random_firing,

            record: Record::new(),
            recording: false
        }
    }

    /// Get network state including the spikes from input neurons
    fn get_full_network_state(&mut self, input: Spikes) -> Spikes {
        let mut state = Spikes::new(self.synapse.neuron_count());

        // Add network state
        state.data.slice_mut(s![(-(self.inputs as i32))..]).assign(&input.data);
        state.data.slice_mut(s![0..self.neurons.len()]).assign(&self.network_state.data);

        state
    }

    fn get_network_input(&mut self, synaptic_input: &Array1<f32>) -> Array1<f32> {
        // Get input only for network neurons
        let mut network_input = synaptic_input.slice(s![0..self.neurons.len()]).to_owned();

        if let Some(n) = self.noise_input {
            network_input += &(random::random_vector(self.neurons.len(), Uniform::new(n.0, n.1)));
        }

        network_input
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
