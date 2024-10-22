//! Task where the goal is discriminate between N different patterns.
//!

use crate::{Task, TaskEval, TaskInput, TaskOutput, TaskState, TaskEnvironment};

use utils::random;
use utils::math;
use utils::encoding::rate_encode;

use ndarray::{Array, Array1, Array2, Axis};


const N_TRIALS: usize = 16;

const SEND_TIME: u32 = 50;
const RESPONSE_DELAY: u32 = 16;
const RESPONSE_WINDOW: u32 = 100;

const RESPONSE_START_T: u32 = SEND_TIME + RESPONSE_DELAY;
const MAX_T: u32 = SEND_TIME + RESPONSE_DELAY + RESPONSE_WINDOW;

const N_CLASSES: usize = 3;
const PATTERN_SIZE: usize = 5;

const OUTPUTS_PER_CLASS: usize = 5;
const AGENT_INPUTS: usize = PATTERN_SIZE.pow(2);
const AGENT_OUTPUTS: usize = N_CLASSES * OUTPUTS_PER_CLASS;

const PATTERN_MAX_PROBABILITY: f32 = 0.3;

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

pub const PATTERNS: [[f32;25]; 5] = [
[0.0011561034, 0.06622559, 0.028740074, 0.038053624, 0.93971705,
0.02895607, 0.020980915, 0.2684978, 0.074474305, 0.81641483,
0.07613335, 0.03424606, 0.0040422105, 0.011101845, 0.7176192,
0.32731244, 0.937556, 0.16226645, 0.6776351, 0.9081723,
0.73409295, 0.9865458, 0.9638059, 0.9442252, 0.9574313],

[0.9953399, 0.97412777, 0.70939493, 0.7137156, 0.92732006,
0.79044336, 0.8728952, 0.20001277, 0.35282508, 0.95879966,
0.86484444, 0.12602703, 0.23220628, 0.19903651, 0.026103357,
0.8509064, 0.060482394, 0.4071675, 0.5829529, 0.007789653,
0.8824559, 0.7899793, 0.052196264, 0.00951018, 0.0077034584],

[0.5905768, 0.22691146, 0.060718138, 0.06451979, 0.010411645,
0.29749846, 0.05148896, 0.0101640625, 0.056540046, 0.13277686,
0.03487958, 0.5345852, 0.0022553147, 0.05475845, 0.2142955,
0.23426689, 0.011564345, 0.050734133, 0.38911572, 0.14685844,
0.0416429, 0.33551505, 0.03780432, 0.3593929, 0.6682871],

[0.42095488, 0.24399209, 0.9722555, 0.6680904, 0.92513806,
0.16057067, 0.62162894, 0.2521128, 0.8902818, 0.96078885,
0.24955904, 0.39319122, 0.025997335, 0.7766625, 0.9592295,
0.009464657, 0.051567703, 0.8994748, 0.3586955, 0.4383101,
0.0043404484, 0.015370565, 0.050364316, 0.40643016, 0.14189482],

[0.9316442, 0.29768023, 0.12711792, 0.008572187, 0.012528594,
0.8123459, 0.17416883, 0.034686103, 0.0054704887, 0.13582532,
0.9238674, 0.74229044, 0.09661883, 0.6430791, 0.19044618,
0.99144614, 0.372423, 0.2602999, 0.25117347, 0.5847933,
0.75281626, 0.9418608, 0.43188825, 0.5999814, 0.24077879]
];
