use ndarray::{s, Array};


use tasks::cognitive_task::{CognitiveTask, TaskResult, TaskInput, TaskState};

use crate::evolution::phenotype::Phenotype;
use crate::synapses::Synapses;
use crate::model::{NeuronModel,Spikes};


const SYNAPTIC_INPUT_SCALING: f32 = 15.0;


pub fn execute<T: CognitiveTask> (phenotype: &mut Phenotype, mut task: T) -> TaskResult {
    let synapse_size = phenotype.synapses.neuron_count();
    let network_size = phenotype.neurons.size();


    let mut synapse_spikes = Spikes::new(synapse_size);
    let mut network_state = Spikes::new(network_size);

    let mut task_inputs: Vec<TaskInput> = vec![];

    loop {
        let task_state = task.tick(&task_inputs);
        task_inputs.clear();

        if let Some(r) = task_state.result {
            return r;
        }

        // parse sensor data into network input
        // TODO: this should probably be changed in the task, so its sensors give spike output
        let sensordata = task_state.sensor_data.mapv(|x| if x != 0.0 {1.0} else {0.0});

        // Assign sensor input and previous state to the the spike input for synapses
        synapse_spikes.data.slice_mut(s![(-phenotype.inputs)..]).assign(&sensordata);
        synapse_spikes.data.slice_mut(s![0..network_size]).assign(&network_state.data);

        let mut synaptic_input = phenotype.synapses.step(&synapse_spikes) * SYNAPTIC_INPUT_SCALING;

        // Get input for network neurons only
        let network_input = synaptic_input.slice(s![0..network_size]).to_owned();

        network_state = phenotype.neurons.step(network_input);

        // Parse the firing state of output neurons to commands
        for i in 0..phenotype.outputs {
            // TODO: unnecessary cast; make phenotype.outputs be usize
            if network_state.data[i as usize] == 1.0 {
                task_inputs.push(TaskInput { input_id: i });
            }
        }
    }
}
