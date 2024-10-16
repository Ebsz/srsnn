//! The agent is presented with two patterns, and is to decide whether the patterns are similar or not.
//! Output neurons are to fire only if the patterns are different.
//!
//! Two types of trial:
//! - Both patterns are drawn from the same distribution; output neurons should not fire.
//! - Patterns are drawn from different distributions; output neurons should fire.
//!
//! NOTE: The fitness function is currently not working, with a bug that returns perfect fitness
//! for models who do not spike at all.


use crate::{Task, TaskEval, TaskInput, TaskOutput, TaskState, TaskEnvironment};

use utils::random;
use utils::math;
use utils::encoding::rate_encode;

use ndarray::{array, Array, Array1, Array2};
use ndarray_rand::rand_distr::StandardNormal;


const DATASET_SIZE: usize = 128;

const PATTERN_SIZE: usize = 5;
const PATTERN_MAX_PROBABILITY: f32 = 0.5;

const SEND_TIME: u32 = 50;        // How long each pattern is sent.
const SEND_DELAY: u32 = 70;       // Delay between patterns

const RESPONSE_WINDOW: u32 = 50;
const RESPONSE_START_T: u32 = 2 * SEND_TIME + 2 * SEND_DELAY; // t0 for the response window

const MAX_T: u32 = 2 * SEND_TIME + 2 * SEND_DELAY + RESPONSE_WINDOW;

const AGENT_INPUTS: usize = PATTERN_SIZE.pow(2);
const AGENT_OUTPUTS: usize = 25;


#[derive(Debug)]
pub struct PatternTaskResult {
    pub output: Array2<u32>,
    pub is_same: bool
}

#[derive(Clone)]
pub struct PatternTaskSetup {
    pub dist1: Array1<f32>,
    pub dist2: Array1<f32>,
    pub is_same: bool
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
                    output: self.response.clone(),
                    is_same: self.setup.is_same
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
        let data = if self.t < SEND_TIME { // First pattern
            rate_encode(&self.setup.dist1)
        } else if self.t > (SEND_TIME + SEND_DELAY) && self.t < (2*SEND_TIME + SEND_DELAY) { // Second pattern
            rate_encode(&self.setup.dist2)
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

        for _ in 0..DATASET_SIZE {
            //let (s1, s2) = pattern::setup_pair(PATTERN_SIZE);
            //setups.push(s1);
            //setups.push(s2);
            setups.push(pattern::generate(PATTERN_SIZE));
        }

        setups
    }

    fn fitness(results: Vec<Self::Result>) -> f32 {
//        let mut fitness = 0.0;
//
//        for r in &results {
//            let (w, h) = (r.output.shape()[0], r.output.shape()[1]);
//
//            let avg_fr = r.output.iter().sum::<u32>() as f32 /  (w * h)  as f32;
//
//            // p[0]: patterns are equal, p[1]: patterns not equal
//            let prediction = array![1.0 - avg_fr, avg_fr];
//
//            let expected = if r.is_same { array![1.0, 0.0] } else { array![0.0, 1.0] };
//
//            //let ce_loss = math::ml::cross_entropy(&prediction, &expected);
//            let ce_loss = math::ml::cross_entropy(&prediction, &(expected * 0.4));
//
//            //println!("{ce_loss}");
//
//            //println!("avg_fr: {avg_fr}, prediction: {prediction}, expected: {expected}, loss: {ce_loss}");
//
//            fitness += 10.0 - math::minf(&[ce_loss, 10.0]);
//        }
//
//        // Normalize fitness to (0, 100)
//        fitness = fitness * 100.0 /(5.0 * results.len() as f32);
//
//        //if fitness == 100.0 {
//        //    for r in &results {
//        //        println!("is_same: {}", r.is_same);
//        //        println!("{:#?}", r.output);
//        //    }
//
//        //}
//
//        fitness

        Self::accuracy(&results).unwrap()
    }

    fn accuracy(results: &[Self::Result]) -> Option<f32> {
        let mut correct = 0;

        for r in results {
            let (w, h) = (r.output.shape()[0], r.output.shape()[1]);

            let avg_fr = r.output.iter().sum::<u32>() as f32 /  (w * h)  as f32;

            // p[0]: patterns are equal, p[1]: patterns not equal
            let prediction = array![1.0 - avg_fr, avg_fr];

            let is_same = math::max_index(prediction) == 0;

            if is_same == r.is_same {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / results.len() as f32;

        Some(accuracy)
    }
}

//pub fn generate_setup(n) -> PatternTaskSetup {
//    if random:.random_range((0.0,1.0)) < 0.5 {
//        let d = pattern(n, (random::random_range((0,n)), random::random_range((0,n))));
//
//        // First setup has equal patterns
//        PatternTaskSetup {
//            dist1: d.clone(),
//            dist2: d,
//            is_same: true}
//    } else {
//
//    }
//
//}

pub mod pattern {
    use super::*;

    pub fn generate(n:usize) -> PatternTaskSetup {
        if rand::random::<f32>() > 0.5 {
            generate_different(n)
        } else {
            generate_equal(n)
        }
    }

    // Creates two setups: one with equal patterns and one with different patterns
    pub fn setup_pair(n: usize) -> (PatternTaskSetup, PatternTaskSetup) {
        (generate_equal(n), generate_different(n))
    }


    // Generate setup with equal patterns
    fn generate_equal(n: usize) -> PatternTaskSetup {
        let d = pattern(n, (random::random_range((0,n)), random::random_range((0,n))));

        PatternTaskSetup {
            dist1: d.clone(),
            dist2: d,
            is_same: true}
    }

    // Generate setup with different patterns
    fn generate_different(n: usize) -> PatternTaskSetup {
        let p1 = (random::random_range((0,n)), random::random_range((0,n)));

        let mut p2;
        loop {
            p2 = (random::random_range((0,n)), random::random_range((0,n)));

            if dist(p1,p2) >= 2.0 {
                break;
            }
        }

        PatternTaskSetup {
            dist1: pattern(n, p1) * 0.3,
            dist2: pattern(n, p2) * 0.3,
            is_same: false}
    }

    fn dist(p1: (usize, usize), p2: (usize, usize)) -> f32 {
        let (x1, y1) = p1;
        let (x2, y2) = p2;

        ((x1 as f32 - x2 as f32).powf(2.0) + (y1 as f32 - y2 as f32).powf(2.0)).sqrt()
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
}
