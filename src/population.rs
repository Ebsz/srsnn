use crate::model::{NeuronModel, FiringState};
use crate::synapses::Synapses;
use crate::record::{Record, RecordType, RecordDataType};

use ndarray::{s, Array2};

/// A population contains a number of neurons of a set model(M), 
/// connected via a specific type of synapses(S)k
pub trait Population<M: NeuronModel, S: Synapses> { 
    
    fn run(&mut self, steps: usize, input: Array2<f32>, record: &mut Record) {
        let n_neurons = self.model().potentials().shape()[0];
        assert!(input.shape() == [steps, n_neurons]);

        let mut firing_state = FiringState::new(n_neurons);

        for i in 0..steps {
            let it = input.slice(s![i, ..]).to_owned(); // External input

            let si = self.synapses().step(firing_state); // Synaptic input

            let step_input = si+&it;

            firing_state = self.model().step(step_input);

            record.log(RecordType::Potentials, RecordDataType::Potentials(self.model().potentials()));
            record.log(RecordType::Spikes, RecordDataType::Spikes(firing_state.state.clone()));
        }
    }

    fn model(&mut self) -> &mut dyn NeuronModel;
    fn synapses(&mut self) ->&mut dyn Synapses;
}
