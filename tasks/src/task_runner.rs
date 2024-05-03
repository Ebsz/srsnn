//! Executes a task with input from a Runnable

use crate::{Task, TaskInput};

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

pub struct TaskRunner<'a, T: Task, R: Runnable> {
    pub task: T,
    pub state: ExecutionState,

    runnable: &'a mut R,
    task_inputs: Vec<TaskInput>,
}

impl<'a, T: Task, R: Runnable> TaskRunner<'a, T, R>
{
    pub fn new(task: T, runnable: &'a mut R) -> TaskRunner<T, R> {
        TaskRunner {
            task,
            state: ExecutionState::INITIALIZED,
            runnable,
            task_inputs: Vec::new()
        }
    }

    /// Executes the task by repeatedly stepping until the task is finished
    pub fn run(&mut self) -> T::Result {
        loop {
            let result = self.step();
            if let Some(r) = result {
                return r;
            }
        }
    }

    pub fn step(&mut self) -> Option<T::Result> {
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
