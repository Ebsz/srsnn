use crate::model::{NeuronModel, FiringState};
use crate::synapses::Synapses;
use crate::record::{Record, RecordType, RecordDataType};

use ndarray::{s, Array2};

/// A population contains a number of neurons of a set model(M), 
/// connected via a specific type of synapses(S)
pub trait Network<M: NeuronModel, S: Synapses> {
    
    fn run(&mut self, steps: usize, input: Array2<f32>, record: &mut Record) {
        let n_neurons = self.model().potentials().shape()[0];
        assert!(input.shape() == [steps, n_neurons]);

        let mut firing_state = FiringState::new(n_neurons);

        for i in 0..steps {
            let external_input = input.slice(s![i, ..]).to_owned();

            firing_state = self.step(external_input, firing_state);

            record.log(RecordType::Potentials, RecordDataType::Potentials(self.model().potentials()));
            record.log(RecordType::Spikes, RecordDataType::Spikes(firing_state.state.clone()));
        }
    }

    fn step(&mut self, input: Array1<f32>, state: FiringState) -> FiringState {
        let synaptic_input = self.synapses().step(state);
        let step_input = synaptic_input + &input;

        self.model().step(step_input)
    }

    fn model(&mut self) -> &mut dyn NeuronModel;
    fn synapses(&mut self) ->&mut dyn Synapses;
}
