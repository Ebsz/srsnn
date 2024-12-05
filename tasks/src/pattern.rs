//! Task where the goal is discriminate between N different patterns.
//!

use crate::{Task, TaskEval, TaskInput, TaskOutput, TaskState, TaskEnvironment};

use utils::math;
use utils::encoding::rate_encode;

use ndarray::{Array, Array1, Array2, Axis};


const N_TRIALS: usize = 128;

const SEND_TIME: u32 = 50;
const RESPONSE_DELAY: u32 = 10;
const RESPONSE_WINDOW: u32 = 100;

const RESPONSE_START_T: u32 = SEND_TIME + RESPONSE_DELAY;
const MAX_T: u32 = SEND_TIME + RESPONSE_DELAY + RESPONSE_WINDOW;

const N_CLASSES: usize = 3;

const OUTPUTS_PER_CLASS: usize = 5;
const AGENT_INPUTS: usize = 4;
const AGENT_OUTPUTS: usize = N_CLASSES * OUTPUTS_PER_CLASS;

const PATTERN_MAX_PROBABILITY: f32 = 0.4;

// Fitness
const MAX_SPIKERATE: f32 = 0.15; // Output spike frequency is capped above this

const MAX_SPIKE_COUNT: usize = RESPONSE_WINDOW as usize * OUTPUTS_PER_CLASS;
const SPIKE_CAP: f32 = MAX_SPIKE_COUNT as f32 * MAX_SPIKERATE;


#[derive(Debug)]
pub struct PatternTaskResult {
    pub response: Array2<u32>,
    pub label: usize
}

#[derive(Clone)]
pub struct PatternTaskSetup {
    pub dist: Array1<f32>,
    pub label: usize
}

pub struct PatternTask {
    setup: PatternTaskSetup,
    t: u32,

    response: Array2<u32>
}

impl Task for PatternTask {
    type Setup = PatternTaskSetup;
    type Result = PatternTaskResult;

    fn new(setup: &Self::Setup) -> Self {
        PatternTask {
            setup: setup.clone(),
            t: 0,
            response: Array::zeros((RESPONSE_WINDOW as usize, AGENT_OUTPUTS))
        }
    }

    fn tick(&mut self, input: TaskInput) -> TaskState<Self::Result> {
        if self.t >= MAX_T {
            return TaskState {
                output: self.get_output(),
                result: Some(PatternTaskResult {
                    response: self.response.clone(),
                    label: self.setup.label
                })
            }
        }

        // Capture response
        if self.t >= MAX_T - RESPONSE_WINDOW {
            self.save_response(&input.data);
        }

        self.t += 1;

        TaskState {
            output: self.get_output(),
            result: None
        }
    }

    fn reset(&mut self) {
        self.t = 0;
        self.response = Array::zeros((RESPONSE_WINDOW as usize, AGENT_INPUTS));
    }

    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: AGENT_INPUTS,
            agent_outputs: AGENT_OUTPUTS
        }
    }
}

impl PatternTask {
    fn get_output(&self) -> TaskOutput {
        let data = if self.t < SEND_TIME {
            rate_encode(&self.setup.dist)
        } else {
            Array::zeros(AGENT_INPUTS)
        };

        TaskOutput { data }
    }

    fn save_response(&mut self, input: &[u32]) {
        let t = (self.t - RESPONSE_START_T) as usize;

        for i in input {
            self.response[[t, *i as usize]] = 1;
        }
    }
}

impl TaskEval for PatternTask {
    fn eval_setups() -> Vec<Self::Setup> {
        let mut setups = vec![];

        // 1. Static patterns
        for _ in 0..N_TRIALS {
            for i in 0..N_CLASSES {
                let dist: Array1<f32> = Array::from_iter(PATTERNS[i].iter())
                    .mapv(|x| *x) * PATTERN_MAX_PROBABILITY;

                setups.push(PatternTaskSetup {
                    dist,
                    label: i,
                });
            }
        }

        setups
    }

    fn fitness(results: Vec<Self::Result>) -> f32 {
        let mut total_loss = 0.0;

        //println!("spike_cap: {}", SPIKE_CAP);

        for r in &results {
            let sum = r.response.sum_axis(Axis(0));
            //println!("{}", sum);

            let mut class_sum: Array1<f32> = sum.exact_chunks(OUTPUTS_PER_CLASS)
                .into_iter().map(|x| x.sum() as f32).collect();
            //println!("before cap: {class_sum}");

            if class_sum.iter().all(|x| *x == 0.0) {
                total_loss += 5.0;
            }

            class_sum = class_sum.mapv(|x| math::minf(&[x, SPIKE_CAP]));
            //println!("after cap: {class_sum}");

            class_sum = &class_sum - math::maxf(class_sum.as_slice().unwrap());
            //println!("after sub: {class_sum}");

            let predictions = math::ml::softmax(&class_sum);
            //println!("prediction: {predictions}");

            let mut label = Array::zeros(N_CLASSES);
            label[r.label] = 1.0;
            //println!("label: {label}");

            let loss = math::ml::cross_entropy(&predictions, &label);

            //println!("loss: {loss}\n");
            total_loss += loss;
        }

        100.0 - total_loss
    }

    fn accuracy(results: &[Self::Result]) -> Option<f32> {
        let mut correct = 0;

        for r in results {
            // Sum spikes per neuron
            let sum = r.response.sum_axis(Axis(0));

            // Sum spikes per class
            let mut class_sum: Array1<f32> = sum.exact_chunks(OUTPUTS_PER_CLASS)
                .into_iter().map(|x| x.sum() as f32).collect();

            // Apply spike cap
            class_sum = class_sum.mapv(|x| math::minf(&[x, SPIKE_CAP]));

            // Subract max count
            class_sum = &class_sum - math::maxf(class_sum.as_slice().unwrap());

            let predictions = math::ml::softmax(&class_sum);

            //println!("pred: {predictions}");

            let predicted_label = math::max_index(predictions);
            //println!("predicted_label: {predicted_label}, label: {}", r.label);

            if predicted_label == r.label {
                correct += 1;
            }
        }

        log::debug!("correct: {correct}/{}", results.len());
        let accuracy = correct as f32 / results.len() as f32;

        Some(accuracy)
    }
}

pub const PATTERNS: [[f32;4]; 3] = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0]
];
