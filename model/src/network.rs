pub mod representation;
pub mod builder;

use crate::neuron::NeuronModel;
use crate::spikes::Spikes;
use crate::synapse::Synapse;
use crate::record::{Record, RecordType};

use utils::environment::Environment;

use ndarray::{s, Array1};

use std::ops::AddAssign;


/// The effect of a spike is multiplied by this; determines the impact of a single spike.
const DEFAULT_SYNAPTIC_COEFFICIENT: f32 = 1.0;

pub trait Network {
    fn step(&mut self, input: Spikes) -> Spikes;

    fn reset_state(&mut self);
}

/// A runnable spiking network, defined by a NeuronModel and a Synapse
pub struct SpikingNetwork<N: NeuronModel, S: Synapse> {
    pub neurons: N,
    pub synapse: S,
    pub env: Environment,

    pub input_synapse: S,

    pub record: Record,
    pub recording: bool,

    pub synaptic_coefficient: f32,

    network_state: Spikes,
}

impl<N: NeuronModel, S: Synapse> Network for SpikingNetwork<N, S> {
    /// Forwards the state of the network a single step.
    fn step(&mut self, input: Spikes) -> Spikes {
        assert!(input.len() == self.env.inputs);

        let external_input = self.input_synapse.step(&input);

        let mut synaptic_input = self.synapse.step(&self.network_state);

        let n_in = external_input.shape()[0];

        synaptic_input.slice_mut(s![..n_in]).add_assign(&external_input);
        synaptic_input = synaptic_input * self.synaptic_coefficient;

        log::trace!("external:{external_input}, synaptic_input: {synaptic_input}, state: {}", self.network_state);

        self.network_state = self.neurons.step(synaptic_input.clone());

        let state: Array1<u32> = (&self.network_state).into();

        let output: Spikes = state.slice(s![-(self.env.outputs as i32)..]).to_owned().into();

        if self.recording {
            self.record.log(RecordType::Potentials, self.neurons.potentials());
            self.record.log(RecordType::Spikes, (&self.network_state).into());
            self.record.log(RecordType::InputSpikes, (&input).into());
            self.record.log(RecordType::OutputSpikes, (&output).into());
            self.record.log(RecordType::SynapticCurrent, synaptic_input);
        }

        output
    }

    fn reset_state(&mut self) {
        self.neurons.reset();
        self.synapse.reset();

        self.network_state = Spikes::new(self.neurons.len());

        self.record = Record::new();
    }
}

impl<N: NeuronModel, S: Synapse> SpikingNetwork<N, S> {
    pub fn new(neurons: N, synapse: S, input_synapse: S, env: Environment) -> SpikingNetwork<N, S> {
        let network_size = neurons.len();

        assert!(input_synapse.shape() == [network_size - env.outputs, env.inputs].into(),
        "Input matrix has wrong shape");

        SpikingNetwork {
            neurons,
            synapse,
            env,

            input_synapse,

            network_state: Spikes::new(network_size),

            synaptic_coefficient: DEFAULT_SYNAPTIC_COEFFICIENT,

            record: Record::new(),
            recording: false
        }
    }

    pub fn enable_recording(&mut self) {
        self.recording = true;
    }
}
