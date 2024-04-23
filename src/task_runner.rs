use crate::phenotype::Phenotype;

use tasks::{Task, TaskResult, TaskInput};

use ndarray::Array1;


#[derive(PartialEq)]
pub enum ExecutionState {
    INITIALIZED,
    RUNNING,
    FINISHED,
}

pub trait Runnable {
    fn step(&mut self, sensors: Array1<f32>) -> Vec<TaskInput>;
    fn reset(&mut self);
}

/// Runs a network on task
pub struct TaskRunner<'a, T, R: TaskResult>
where
    T: Task<R>,
{
    pub task: T,
    pub state: ExecutionState,

    runnable: &'a mut dyn Runnable,
    task_inputs: Vec<TaskInput>,

    result: Option<R> // NOTE: This is never set, we're required to use R
}

impl<'a, T, R: TaskResult> TaskRunner<'a, T, R>
where
    T: Task<R>
{
    pub fn new(task: T, runnable: &mut dyn Runnable) -> TaskRunner<T, R> {
        TaskRunner {
            task,
            result: None,
            state: ExecutionState::INITIALIZED,
            runnable,
            task_inputs: Vec::new()
        }
    }

    /// Executes the task by repeatedly stepping until the task is finished
    pub fn run(&mut self) -> R {
        loop {
            let result = self.step();
            if let Some(r) = result {
                return r;
            }
        }
    }

    pub fn step(&mut self) -> Option<R>{
        self.state = ExecutionState::RUNNING;

        // Task step
        let task_state = self.task.tick(&self.task_inputs);
        self.task_inputs.clear();

        // If we have a result, return it
        if let Some(r) = task_state.result {
            self.state = ExecutionState::FINISHED;
            return Some(r);
        }

        self.task_inputs = self.runnable.step(task_state.sensor_data);

        None
    }

    /// Reset the execution to its initial state
    pub fn reset(&mut self) {
        self.state = ExecutionState::INITIALIZED;
        self.runnable.reset();
        self.task.reset();
    }
}
