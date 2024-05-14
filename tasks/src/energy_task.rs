use crate::{Task, TaskInput, TaskOutput, TaskState, TaskEnvironment, TaskEval};

use utils::random;

use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;

const DEFAULT_MAX_TIMEOUT: u32 = 10;
const DEFAULT_INPUT_DURATION: u32 =  100;

const AGENT_INPUTS: usize = 3;
const AGENT_OUTPUTS: usize = 1;

#[derive(Copy, Clone)]
pub struct EnergyTaskSetup {
   max_timeout: u32,
   input_duration: u32
}

pub struct EnergyTaskResult {
    t: u32
}

pub struct EnergyTask {
    ticks: u32,
    last_spike: u32,
    setup: EnergyTaskSetup
}

impl Task for EnergyTask {
    type Setup = EnergyTaskSetup;
    type Result = EnergyTaskResult;

    fn new(setup: &Self::Setup) -> Self {

        EnergyTask {
            ticks: 0,
            last_spike: 0,
            setup: *setup
        }
    }

    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState<Self::Result> {
        if input.len() != 0 {
            self.last_spike = self.ticks;
        }

        if self.ticks - self.last_spike > self.setup.max_timeout {
            return TaskState {
                output: TaskOutput { data: Array::zeros(AGENT_INPUTS) },
                result: Some(EnergyTaskResult {t: self.last_spike })
            }
        }

        self.ticks += 1;


        TaskState {
            output: TaskOutput {
                data: match self.ticks < self.setup.input_duration {
                    true => { random::random_vector(AGENT_INPUTS, Uniform::new(0.0, 1.0)) },
                    false => { Array::zeros(AGENT_INPUTS) }
                },
            },
            result: None
        }
    }

    fn reset(&mut self) {
        self.ticks = 0;
        self.last_spike = 0;
    }

    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: AGENT_INPUTS,
            agent_outputs: AGENT_OUTPUTS
        }
    }
}

impl TaskEval for EnergyTask {
    fn eval_setups() -> Vec<EnergyTaskSetup> {
        vec![EnergyTaskSetup {
            max_timeout: DEFAULT_MAX_TIMEOUT,
            input_duration: DEFAULT_INPUT_DURATION
        }]
    }

    fn fitness(results: Vec<EnergyTaskResult>) -> f32 {
        results.iter().fold(0u32, |acc, r| acc + r.t) as f32 / results.len() as f32
    }
}
