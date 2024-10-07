//! Compute the exclusive or function
//!
//! The agent is to give output if it receives spikes on either
//! of its inputs, but not both.

use crate::{Task, TaskInput, TaskOutput, TaskState, TaskEnvironment, TaskEval};

use ndarray::{array, Array};


const AGENT_INPUTS: usize = 2;
const AGENT_OUTPUTS: usize = 1;

const INPUT_T: u32 = 50;
const READOUT_T: u32 = 100;

#[derive(Copy, Clone)]
pub struct XORTaskSetup {
    inputs: (u32, u32),
    output: u32
}

#[derive(Debug)]
pub struct XORTaskResult {
    spike_times: Vec<u32>,
    desired_output: u32
}

pub struct XORTask {
    setup: XORTaskSetup,
    ticks: u32,
    spike_times: Vec<u32>
}

impl Task for XORTask {
    type Setup = XORTaskSetup;
    type Result = XORTaskResult;

    fn new(setup: &Self::Setup) -> Self {
        XORTask {
            setup: *setup,
            ticks: 0,
            spike_times: Vec::new()
        }
    }

    fn tick(&mut self, input: TaskInput) -> TaskState<Self::Result> {
        let output: TaskOutput;

        if self.ticks <= INPUT_T {
            output = self.get_output();
        } else {
            output = TaskOutput {data: Array::zeros(AGENT_INPUTS)};
        }

        if self.ticks > INPUT_T {
            if input.data.len() != 0 {
                self.spike_times.push(self.ticks);
            }
        }

        if self.ticks > INPUT_T + READOUT_T {
            return TaskState {
                output,
                result: Some(XORTaskResult {spike_times: self.spike_times.clone(), desired_output: self.setup.output })
            }
        }

        self.ticks += 1;

        TaskState {
            output,
            result: None
        }
    }

    fn reset(&mut self) {

    }

    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: AGENT_INPUTS,
            agent_outputs: AGENT_OUTPUTS
        }
    }
}

impl XORTask {
    fn get_output(&self) -> TaskOutput {
        TaskOutput {
            data: array![self.setup.inputs.0 as f32, self.setup.inputs.1 as f32]
        }
    }
}

impl TaskEval for XORTask {
    fn eval_setups() -> Vec<XORTaskSetup> {
        vec![
            XORTaskSetup { inputs: (0, 0), output: 0 },
            XORTaskSetup { inputs: (1, 0), output: 1 },
            XORTaskSetup { inputs: (0, 1), output: 1 },
            XORTaskSetup { inputs: (1, 1), output: 0 },
        ]
    }

    fn fitness(results: Vec<XORTaskResult>) -> f32 {
        let mut fitness = 0.0;

        for r in results {
            let network_output = if r.spike_times.len() == 0 { 0 } else { 1 };

            if network_output == r.desired_output {
                fitness += 25.0;
            }
        }

        fitness
    }

    fn accuracy(_results: &[Self::Result]) -> Option<f32> {
        None
    }
}
