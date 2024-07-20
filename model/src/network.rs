pub mod representation;
pub mod builder;
pub mod runnable;

use crate::neuron::NeuronModel;
use crate::spikes::Spikes;
use crate::synapse::Synapse;
use crate::record::{Record, RecordType, RecordDataType};

use utils::random;
use utils::environment::Environment;

use ndarray::{s, Array1, Array2};
use rand::distributions::Uniform;


/// Output of a neuron is multiplied by this; determines the impact of a single spike.
const SYNAPTIC_INPUT_SCALING: f32 = 15.0;

const RANDOM_FIRING_PROBABILITY: f32 = 0.01;

pub trait Network {
    fn step(&mut self, input: Spikes) -> Spikes;

    fn reset_state(&mut self);
}

/// A runnable RSNN.
///
/// A network consists of a NeuronModel of (n - n_inputs) neurons,
/// and a Synapse of n neurons.
///
/// In the synapse, output neurons are the first 0..n_output neurons,
/// and input neurons are the last (n - n_inputs)..n neurons.
pub struct SpikingNetwork<N: NeuronModel, S: Synapse> {
    pub neurons: N,
    pub synapse: S,
    pub env: Environment,

    pub input_matrix: Array2<f32>,

    pub record: Record,
    pub recording: bool,

    network_state: Spikes,

    // Optional energy input
    noise_input: Option<(f32, f32)>,
    random_firing: bool
}

impl<N: NeuronModel, S: Synapse> Network for SpikingNetwork<N, S> {
    fn step(&mut self, input: Spikes) -> Spikes {
        assert!(input.len() == self.env.inputs);

        let external_input = self.input_matrix.dot(&input.as_float());
        let synaptic_input = self.synapse.step(&self.network_state);

        let mut total_input = (synaptic_input + external_input) * SYNAPTIC_INPUT_SCALING;

        if let Some(n) = self.noise_input {
            total_input += &(random::random_vector(self.neurons.len(), Uniform::new(n.0, n.1)));
        }

        self.network_state = self.neurons.step(total_input);

        if self.random_firing {
            self.add_random_output_firing();
        }

        if self.recording {
            self.record.log(RecordType::Potentials, RecordDataType::Potentials(self.neurons.potentials()));
            self.record.log(RecordType::Spikes, RecordDataType::Spikes(self.network_state.as_float()));
        }

        self.network_state.clone()
    }

    fn reset_state(&mut self) {
        self.neurons.reset();
        self.synapse.reset();
    }
}

impl<N: NeuronModel, S: Synapse> SpikingNetwork<N, S> {
    pub fn new(neurons: N, synapse: S, input_matrix: Array2<f32>, env: Environment) -> SpikingNetwork<N, S> {
        let noise_input = None;
        let random_firing = false;

        let network_size = neurons.len();

        SpikingNetwork {
            neurons,
            synapse,
            env,
            input_matrix,

            network_state: Spikes::new(network_size),

            noise_input,
            random_firing,

            record: Record::new(),
            recording: false
        }
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
