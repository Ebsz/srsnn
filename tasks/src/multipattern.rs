use crate::{Task, TaskEval, TaskInput, TaskOutput, TaskState, TaskEnvironment};

use utils::random;
use utils::math;
use utils::encoding::rate_encode;

use ndarray::{array, Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::StandardNormal;


const DATASET_SIZE: usize = 1024;

const CLASS_CENTERS: [(usize, usize); 5] = [(0, 0), (2, 2), (4, 4), (4, 0), (0, 4)];

const N_CLASSES: usize = 5;

const PATTERN_SIZE: usize = 5;
const PATTERN_MAX_PROBABILITY: f32 = 0.2;

const SEND_TIME: u32 = 50;          // How many timesteps to send the pattern
const RESPONSE_DELAY: u32 = 70;     // Delay between pattern send and response window
const RESPONSE_WINDOW: u32 = 50;    // Number of timesteps to record the response

const RESPONSE_START_T: u32 = SEND_TIME + RESPONSE_DELAY;
const MAX_T: u32 = SEND_TIME + RESPONSE_DELAY + RESPONSE_WINDOW;

const AGENT_INPUTS: usize = PATTERN_SIZE.pow(2);

const OUTPUTS_PER_CLASS: usize = 5;
const AGENT_OUTPUTS: usize = N_CLASSES * OUTPUTS_PER_CLASS;


#[derive(Debug)]
pub struct MultiPatternTaskResult {
    pub response: Array2<u32>,
    pub label: usize
}

#[derive(Clone)]
pub struct MultiPatternSetup {
    pub dist: Array1<f32>,
    pub label: usize
}

pub struct MultiPatternTask {
    setup: MultiPatternSetup,
    t: u32,

    response: Array2<u32>
}

impl Task for MultiPatternTask {
    type Setup = MultiPatternSetup;
    type Result = MultiPatternTaskResult;

    fn new(setup: &Self::Setup) -> Self {
        MultiPatternTask {
            setup: setup.clone(),
            t: 0,
            response: Array::zeros((RESPONSE_WINDOW as usize, AGENT_OUTPUTS))
        }
    }

    fn tick(&mut self, input: TaskInput) -> TaskState<Self::Result> {
        if self.t >= MAX_T {
            return TaskState {
                output: self.get_output(),
                result: Some(MultiPatternTaskResult {
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

impl MultiPatternTask {
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

impl TaskEval for MultiPatternTask {
    fn eval_setups() -> Vec<Self::Setup> {
        let mut setups = vec![];

        for _ in 0..DATASET_SIZE {
            let l = random::random_range((0, N_CLASSES));

            setups.push(MultiPatternSetup {
                dist: pattern(PATTERN_SIZE, CLASS_CENTERS[l]),
                label: l,
            });
        }

        setups
    }

    // TODO: Rate regularization: firing rate of MAX is equal to 1.0.
    // Might not be necessary, because it is relative due to softmax?
    fn fitness(results: Vec<Self::Result>) -> f32 {
        let mut fitness = 0.0;

        for r in &results {
            // sum of spikes per output neuron
            let sum = r.response.sum_axis(Axis(0));

            // sum of spikes per label
            let mut label_sum: Array1<f32> = sum.exact_chunks(OUTPUTS_PER_CLASS)
                .into_iter().map(|x| x.sum() as f32).collect();

            assert!(label_sum.shape()[0] == N_CLASSES);

            label_sum = &label_sum - math::maxf(label_sum.as_slice().unwrap());

            // Firing rates per label
            // let firing_rates: Array1<f32> = label_sum.map(|x| *x as f32)
            //    / (OUTPUTS_PER_CLASS as f32 * r.response.shape()[0] as f32);

            //let firing_rates = array![0.0,0.0,0.0,0.0, 2.0];
            //println!("firing rates: {}", firing_rates);

            let predictions = math::ml::softmax(&label_sum);
            let mut label = Array::zeros(N_CLASSES);
            label[r.label] = 1.0;

            let loss = math::ml::cross_entropy(&predictions, &label);

            //println!("label sum: {}", label_sum);
            //println!("predictions: {}", predictions);
            //println!("label: {}", label);
            //println!("loss: {:?}", loss);

            fitness += 10.0 - math::minf(&[loss, 10.0]);
        }

        fitness = fitness * 100.0 / (10.0 * results.len() as f32);

        fitness
    }

    fn accuracy(results: &[MultiPatternTaskResult]) -> Option<f32> {
        let mut correct = 0;

        for r in results {
            let sum = r.response.sum_axis(Axis(0));

            // sum of spikes per label
            let mut label_sum: Array1<f32> = sum.exact_chunks(OUTPUTS_PER_CLASS)
                .into_iter().map(|x| x.sum() as f32).collect();

            assert!(label_sum.shape()[0] == N_CLASSES);

            label_sum = &label_sum - math::maxf(label_sum.as_slice().unwrap());

            let predictions = math::ml::softmax(&label_sum);

            let predicted_label = math::max_index(predictions);

            //println!("p: {predicted_label}, l: {}", r.label);
            if predicted_label == r.label {
                correct += 1;
            }
        }

        //println!("correct: {correct}/{}", results.len());
        let accuracy = correct as f32 / results.len() as f32;

        Some(accuracy)
    }
}

/// Sample [t x len] pattern
pub fn pattern(n: usize, center: (usize, usize)) -> Array1<f32> {
    let (px, py) = center; //(random::random_range((0,n)), random::random_range((0,n)));

    let dist = |(x1,y1), (x2, y2)|
        ((x1 as f32 - x2 as f32).powf(2.0) + (y1 as f32 - y2 as f32).powf(2.0)).sqrt();

    let mut p = Array::zeros((n, n));

    for (ix, v) in p.iter_mut().enumerate() {
        let i = (ix / n) as u32;
        let j = (ix % n) as u32;

        let x = dist((i,j), (px,py)) * 1.5 - 5.0 +  random::random_sample::<f32, _>(StandardNormal) * 1.7;

        *v = math::ml::sigmoid(x) * PATTERN_MAX_PROBABILITY;
    }

    Array::from_iter(p.iter().cloned())
}
