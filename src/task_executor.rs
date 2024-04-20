use crate::phenotype::Phenotype;

use tasks::{Task, TaskResult, TaskInput};

use model::synapse::Synapse;
use model::neuron::NeuronModel;
use model::spikes::Spikes;
use model::record::{Record, RecordType, RecordDataType};

use utils::random;

use ndarray::s;

use rand::distributions::Uniform;


// TODO: This belongs somewhere else
const SYNAPTIC_INPUT_SCALING: f32 = 18.0;

#[derive(PartialEq)]
pub enum ExecutionState {
    INITIALIZED,
    RUNNING,
    FINISHED,
}

/// Executes a network on task
pub struct TaskExecutor<'a, T: Task> {
    pub record: Record,
    pub task: T,
    pub phenotype: &'a mut Phenotype,
    pub state: ExecutionState,
    pub noise_input: Option<(f32, f32)>, // Optionally add random noise to model

    task_inputs: Vec<TaskInput>,
    network_size: usize,
    synapse_spikes: Spikes,
    network_state: Spikes
}

impl<'a, T: Task> TaskExecutor<'a, T> {
    pub fn new(task: T, phenotype: &mut Phenotype) -> TaskExecutor<T> {
        let synapse_size = phenotype.synapse.neuron_count();
        let network_size = phenotype.neurons.size();

        let record: Record = Record::new();

        let noise_input = None;

        TaskExecutor {
            task,
            phenotype,
            record,
            state: ExecutionState::INITIALIZED,
            network_size,
            noise_input,
            task_inputs: Vec::new(),
            synapse_spikes: Spikes::new(synapse_size),
            network_state: Spikes::new(network_size)
        }
    }

    /// Executes the task by repeatedly stepping until the task is finished
    pub fn execute(&mut self, should_record: bool) -> TaskResult{
        loop {
            let result = self.step(should_record);
            if let Some(r) = result {
                return r
            }
        }
    }

    pub fn step(&mut self, should_record: bool) -> Option<TaskResult>{
        self.state = ExecutionState::RUNNING;

        // Task step
        let task_state = self.task.tick(&self.task_inputs);
        self.task_inputs.clear();

        // If we have a result, return it
        if let Some(r) = task_state.result {
            self.state = ExecutionState::FINISHED;
            return Some(r);
        }

        // Assign sensor input and previous state to the the spike input for synapses
        self.synapse_spikes.data.slice_mut(s![(-self.phenotype.inputs)..]).assign(&task_state.sensor_data);
        self.synapse_spikes.data.slice_mut(s![0..self.network_size]).assign(&self.network_state.data);

        // Synapse step
        let synaptic_input = self.phenotype.synapse.step(&self.synapse_spikes) * SYNAPTIC_INPUT_SCALING;

        // Get input only for network neurons
        let mut network_input = synaptic_input.slice(s![0..self.network_size]).to_owned();

        if let Some(n) = self.noise_input {
            network_input += &(random::random_vector(self.network_size, Uniform::new(n.0, n.1)));

        }

        self.network_state = self.phenotype.neurons.step(network_input);

        if should_record {
            self.record.log(RecordType::Potentials, RecordDataType::Potentials(self.phenotype.neurons.potentials()));
            self.record.log(RecordType::Spikes, RecordDataType::Spikes(self.network_state.data.clone()));
        }

        // Parse the firing state of output neurons to commands
        for i in 0..self.phenotype.outputs {
            // TODO: unnecessary cast; make phenotype.outputs be usize
            // If the neurons assigned to be output fire, add the corresponding input
            if self.network_state.data[i as usize] == 1.0 {
                self.task_inputs.push(TaskInput { input_id: i });
            }
        }
        None
    }

    /// Reset the execution to its initial state
    pub fn reset(&mut self) {
        self.state = ExecutionState::INITIALIZED;
        self.phenotype.reset();
        self.task.reset();
    }
}
