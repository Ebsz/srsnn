pub mod sensor;
pub mod task_runner;

pub mod catching_task;
pub mod movement_task;
pub mod survival_task;
pub mod energy_task;
pub mod mnist_task;
pub mod xor_task;
pub mod pole_balancing_task;

use ndarray::Array1;

use std::fmt::Debug;


/// Input to the task, from the agent
#[derive(Debug)]
pub struct TaskInput {
    pub input_id: u32
}

/// Output from the task, to the agent
#[derive(Debug)]
pub struct TaskOutput {
    pub data: Array1<f32>
}

pub struct TaskEnvironment {
    pub agent_inputs: usize,
    pub agent_outputs: usize,
}

#[derive(Debug)]
pub struct TaskState<R> {
    pub result: Option<R>,
    pub output: TaskOutput,
}

pub trait Task {
    type Setup: Clone + Send + Sync;
    type Result: Debug;

    fn new(setup: &Self::Setup) -> Self;
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState<Self::Result>;
    fn reset(&mut self);
    fn environment() -> TaskEnvironment;
}


pub trait TaskEval: Task {
    fn eval_setups() -> Vec<Self::Setup>;

    fn fitness(results: Vec<Self::Result>) -> f32;
}

