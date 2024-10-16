pub mod representation;
pub mod builder;
pub mod runnable;

use crate::neuron::NeuronModel;
use crate::spikes::Spikes;
use crate::synapse::Synapse;
use crate::record::{Record, RecordType, RecordDataType};

use utils::environment::Environment;

use ndarray::{s, Array1, Array2};

use std::ops::AddAssign;


/// The effect of a spike is multiplied by this; determines the impact of a single spike.
const DEFAULT_SYNAPTIC_COEFFICIENT: f32 = 5.0;
//const RANDOM_FIRING_PROBABILITY: f32 = 0.01;

pub trait Network {
    fn step(&mut self, input: Spikes) -> Spikes;

    fn reset_state(&mut self);
}

/// A runnable RSNN, defined by a NeuronModel and a Synapse
pub struct SpikingNetwork<N: NeuronModel, S: Synapse> {
    pub neurons: N,
    pub synapse: S,
    pub env: Environment,

    pub input_matrix: Array2<f32>,

    pub record: Record,
    pub recording: bool,

    pub synaptic_coefficient: f32,

    network_state: Spikes,

    // Optional energy input
    //noise_input: Option<(f32, f32)>,
    //random_firing: bool
}

impl<N: NeuronModel, S: Synapse> Network for SpikingNetwork<N, S> {
    /// Forwards the state of the network a single step.
    fn step(&mut self, input: Spikes) -> Spikes {
        assert!(input.len() == self.env.inputs);

        let external_input = self.input_matrix.dot(&input.as_float());

        let mut synaptic_input = self.synapse.step(&self.network_state);

        let n_in = external_input.shape()[0];

        synaptic_input.slice_mut(s![..n_in]).add_assign(&external_input);
        synaptic_input = synaptic_input * self.synaptic_coefficient;

        //let mut total_input: Array1<f32> = Array::zeros(synaptic_input.raw_dim());

        //println!("{}\n",total_input);


        //let total_input = (synaptic_input + external_input) * self.synaptic_coefficient;

        //if let Some(n) = self.noise_input {
        //    total_input += &(random::random_vector(self.neurons.len(), Uniform::new(n.0, n.1)));
        //}

        self.network_state = self.neurons.step(synaptic_input);

        //if self.random_firing {
        //    self.add_random_output_firing();
        //}

        //self.output_matrix.dot(&state).into();
        let state: Array1<u32> = (&self.network_state).into();

        let output: Spikes = state.slice(s![-(self.env.outputs as i32)..]).to_owned().into();

        if self.recording {
            self.record.log(RecordType::Potentials, RecordDataType::Potentials(self.neurons.potentials()));
            self.record.log(RecordType::Spikes, RecordDataType::Spikes(self.network_state.as_float()));
            self.record.log(RecordType::InputSpikes, RecordDataType::InputSpikes(input.as_float()));
            self.record.log(RecordType::OutputSpikes, RecordDataType::OutputSpikes(output.as_float()));
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
    pub fn new(neurons: N, synapse: S, input_matrix: Array2<f32>, env: Environment) -> SpikingNetwork<N, S> {
        //let noise_input = None;
        //let random_firing = false;

        let network_size = neurons.len();

        SpikingNetwork {
            neurons,
            synapse,
            env,

            input_matrix,

            network_state: Spikes::new(network_size),

            synaptic_coefficient: DEFAULT_SYNAPTIC_COEFFICIENT,

            record: Record::new(),
            recording: false

            //noise_input,
            //random_firing,
        }
    }

    pub fn enable_recording(&mut self) {
        self.recording = true;
    }
    //fn add_random_output_firing(&mut self) {
    //    let random_spikes: Array1<f32> = random::random_vector(self.neurons.len(), Uniform::new(0.0, 1.0))
    //        .mapv(|x| if x > (1.0 - RANDOM_FIRING_PROBABILITY) {1.0} else {0.0});

    //    self.network_state.data = (&self.network_state.as_float() + &random_spikes)
    //        .mapv(|x| if x != 0.0 { true } else { false });
    //}
}
