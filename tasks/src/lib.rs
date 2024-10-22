pub mod sensor;
pub mod task_runner;

use ndarray::Array1;

use std::fmt::Debug;

pub mod testing;

pub mod pattern;

pub mod catching_task;
pub mod pole_balancing_task;

pub mod xor_task;
pub mod lorenz_task;
pub mod mnist_task;

pub mod pattern_similarity;
pub mod multipattern;



/// Input to the task, from the agent
#[derive(Debug)]
pub struct TaskInput {
    pub data: Vec<u32>
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
    fn tick(&mut self, input: TaskInput) -> TaskState<Self::Result>;
    fn reset(&mut self);
    fn environment() -> TaskEnvironment;
}

pub trait TaskEval: Task {
    fn eval_setups() -> Vec<Self::Setup>;

    fn fitness(results: Vec<Self::Result>) -> f32;

    fn accuracy(results: &[Self::Result]) -> Option<f32>;
}
