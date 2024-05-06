use crate::neuron::NeuronModel;
use crate::spikes::Spikes;
use crate::synapse::Synapse;
use crate::record::{Record, RecordType, RecordDataType};

use ndarray::{s, Array1, Array2};
use std::time::Instant;


// Output of a neuron is multiplied by this; determines the impact of a single spike.
const SYNAPTIC_INPUT_SCALING: f32 = 15.0;


/// A network contains a neurons of a specific model M that are
/// connected via a specific type of synapse S
pub trait Network<M: NeuronModel, S: Synapse> {

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

    fn step(&mut self, input: Array1<f32>, state: Spikes) -> Spikes {
        let synaptic_input = self.synapse().step(&state) * SYNAPTIC_INPUT_SCALING;

        let step_input = synaptic_input + &input;

        self.model().step(step_input)
    }

    fn model(&mut self) -> &mut M;
    fn synapse(&mut self) -> &mut S;
}
