use crate::{Task, TaskEval, TaskInput, TaskOutput, TaskState, TaskEnvironment};

use utils::math::ml;
use utils::encoding;
use utils::math;

use utils::analysis;

use ndarray::{s, Array, Array1, Array2, Axis};


const AGENT_INPUTS: usize = 16;
const AGENT_OUTPUTS: usize = 8;

const MAX_T: u32 = 1000;

#[derive(Copy, Clone)]
pub struct TestTaskSetup {

}

#[derive(Debug)]
pub struct TestTaskResult {
    pub record: Array2<u32>,
}

pub struct TestTask {
    setup: TestTaskSetup,

    record: Array2<u32>,
    t: u32,
}

impl Task for TestTask {
    type Setup = TestTaskSetup;
    type Result = TestTaskResult;

    fn new(setup: &Self::Setup) -> Self {
        TestTask {
            setup: *setup,

            record: Array::zeros((MAX_T as usize, AGENT_OUTPUTS)),
            t: 0
        }
    }

    fn tick(&mut self, input: TaskInput) -> TaskState<Self::Result> {
        if self.t >= MAX_T {
            return TaskState {
                output: self.get_output(),
                result: Some(TestTaskResult {
                    record: self.record.clone(),
                })
            }
        }

        self.save_response(&input.data);

        self.t += 1;

        TaskState {
            output: self.get_output(),
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

impl TaskEval for TestTask {
    fn eval_setups() -> Vec<TestTaskSetup> {
        vec![ TestTaskSetup { } ]
    }

    fn fitness(results: Vec<TestTaskResult>) -> f32 {
        // 1. Compute firing rate
        // 2. Compute function from firing rate
        // 3. Compare the difference between the function and the target function

        for r in results {
            let fr = analysis::firing_rate(r.record, 32);

            //println!("{}", fr);

            //println!("max: {}", math::maxf(&fr.clone().into_iter().collect::<Vec<f32>>()));
            //println!("{:#?}", fr);
            //println!("fr max: {}", math::maxf(fr.as_slice().unwrap()));
        }

        let f = rand::random();

        f
    }

    fn accuracy(_results: &[Self::Result]) -> Option<f32> {
        None
    }
}

impl TestTask {
    fn get_output(&self) -> TaskOutput {
        TaskOutput {
            data: output(self.t)
        }
    }

    fn save_response(&mut self, input: &[u32]) {
        //println!("t: {}", self.t);
        //println!("input: {:?}", input);
        for i in input {
            self.record[[self.t as usize, *i as usize]] = 1;
        }
    }
}

fn output(t: u32) -> Array1<f32> {
    let o = encoding::rate_encode(&ff(t));

    //println!("{}", o);

    o
}

fn f(t: u32) -> f32 {
    0.0
}

fn ff(t: u32) -> Array1<f32> {
    Array::ones(AGENT_INPUTS) * ml::sigmoid(2.0 * f32::sin(0.04 * t as f32) - 4.0)
}
