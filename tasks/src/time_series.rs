//! Time series prediction task
//!
//! Given a t x n time series, the goal is for the network to replicate the values at each
//! timestep.

use ts::TimeSeries;

use crate::{Task, TaskEval, TaskInput, TaskOutput, TaskState, TaskEnvironment};
use crate::lorenz_task;

use utils::math;
use utils::encoding::rate_encode;

use ndarray::{s, Array, Array1, Array2, Axis};
use ndarray_rand::rand::Rng;


const N_SETUPS: usize = 100;

const MAX_FR: f32 = 0.2;

const T: u32 = 800;

const SEND_TIME: u32 = 300; // How long the time series is sent for

const N_VARIABLES: usize = 1; //TODO: TMP, for sin series
const OUTPUTS_PER_VARIABLE: usize = 10;

const AGENT_INPUTS: usize = N_VARIABLES * OUTPUTS_PER_VARIABLE;
const AGENT_OUTPUTS: usize = N_VARIABLES * OUTPUTS_PER_VARIABLE;

const OUTPUT_MAX_PROBABILITY: f32 = 0.1;

#[derive(Debug)]
pub struct TimeSeriesResult {
    pub response: Array2<u32>,
    pub observed: Array2<f32>
}

#[derive(Clone)]
pub struct TimeSeriesTaskSetup {
    pub series: Array2<f32>,
}

pub struct TimeSeriesTask<S: TimeSeries>  {
    setup: TimeSeriesTaskSetup,
    t: u32,

    response: Array2<u32>,

    s: Option<S>
}

impl<S: TimeSeries> Task for TimeSeriesTask<S> {
    type Setup = TimeSeriesTaskSetup;
    type Result = TimeSeriesResult;

    fn new(setup: &Self::Setup) -> Self {
        TimeSeriesTask {
            setup: setup.clone(),
            t: 0,
            response: Array::zeros((T as usize, AGENT_OUTPUTS)),
            s: None
        }
    }

    fn tick(&mut self, input: TaskInput) -> TaskState<Self::Result> {
        if self.t >= T {
            return TaskState {
                output: self.get_output(),
                result: Some(TimeSeriesResult {
                    response: self.response.clone(),
                    observed: self.setup.series.clone()
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
        self.t = 0;
        self.response = Array::zeros((T as usize, AGENT_INPUTS));
    }

    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: AGENT_INPUTS, //NEURONS_PER_VAR * S::variables(),
            agent_outputs: AGENT_OUTPUTS, //NEURONS_PER_VAR * S::variables()
        }
    }
}


impl<S: TimeSeries> TimeSeriesTask<S> {
    fn get_output(&self) -> TaskOutput {
        let data = if self.t < SEND_TIME {
            let mut a = Array::ones(AGENT_INPUTS);

            a = a * self.setup.series[[self.t as usize, 0]];

            rate_encode(&(a * OUTPUT_MAX_PROBABILITY))
        } else {
            Array::zeros(AGENT_INPUTS)
        };

        TaskOutput { data }
    }

    fn save_response(&mut self, input: &[u32]) {
        let t = self.t as usize;

        for i in input {
            self.response[[t, *i as usize]] = 1;
        }
    }
}

impl<S: TimeSeries> TaskEval for TimeSeriesTask<S> {
    fn eval_setups() -> Vec<Self::Setup> {
        let mut setups = vec![];

        for _ in 0..N_SETUPS {
            // Generate a time-series
            //let mut s: Array2<f64> = lorenz_task::simulate(T as usize, lorenz_task::CHAOS, (1.0, 1.0, 1.2));
            let mut s: Array2<f64> = S::generate(T as usize);

            // Normalize to (0,1) using the min/max of all values, which preserves the relative values
            let min = math::minf(s.as_slice().unwrap());
            let max = math::maxf(s.as_slice().unwrap());

            s = (&s - min) / (max- min);

            let min = math::minf(s.as_slice().unwrap());
            let max = math::maxf(s.as_slice().unwrap());

            //println!("min: {min}, max: {max}");
            //println!("mean: {}, std: {}", s.mean().unwrap(), s.std(0.0));

            setups.push(TimeSeriesTaskSetup { series: s.mapv(|x| x as f32) });
        }

        setups
    }

    fn fitness(results: Vec<Self::Result>) -> f32 {
        // The window of calculating firing rate; determines how "sharp" it is
        const FR_WINDOW: usize = 20;

        // Time delay after SEND_TIME before evaluating.
        const EVAL_DELAY: u32 = 75;

        const EVAL_T: usize = (T - SEND_TIME - EVAL_DELAY) as usize;

        let mut fitness = 0.0;

        let n_results = results.len();

        for r in results {
            let firing_rates: Array2<f32> = utils::analysis::firing_rate(r.response, FR_WINDOW);

            //let fr = firing_rates.mapv(|x| math::minf(&[x, MAX_FR])) * (1.0 / MAX_FR);
            let fr = firing_rates;

            // Average firing rate per time per bin
            let mut arr: Array2<f32> = Array::zeros((T as usize, N_VARIABLES));
            for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
                let a: Array1<f32> = fr.slice(s![i,..]).exact_chunks(OUTPUTS_PER_VARIABLE).into_iter()
                    .map(|x| x.sum() / OUTPUTS_PER_VARIABLE as f32 ).collect();

                row.assign(&a);
            }

            // apply max max firing rate per bin and normalize to (0, 1)
            arr = arr.mapv(|x| math::minf(&[x, MAX_FR])) * (1.0 / MAX_FR);

            let predicted: Array2<f32> = arr.slice(s![EVAL_T..,..]).to_owned();
            let observed: Array2<f32> = r.observed.slice(s![EVAL_T..,..]).to_owned();

            assert!(predicted.shape() == observed.shape());

            // Calculate RMS deviations
            let mut deviations: Array2<f32> = Array::zeros(observed.raw_dim());

            for (i, mut row) in deviations.axis_iter_mut(Axis(0)).enumerate() {
                let a: Array1<f32> = (&observed.slice(s![i,..])
                    - &predicted.slice(s![i,..])).map(|x| x.powf(2.0));

                row.assign(&a);
            }

            fitness += - deviations.sum();
        }

        fitness /= n_results as f32;

        fitness
    }

    fn accuracy(results: &[Self::Result]) -> Option<f32> {
        None
    }
}

pub mod ts {
    use super::*;

    pub trait TimeSeries {
        fn generate(steps: usize) -> Array2<f64>;
        fn variables() -> usize;
    }

    pub struct LorenzSystem;
    impl TimeSeries for LorenzSystem {

        fn generate(steps: usize) -> Array2<f64> {
            let init_values = (1.0,1.0,1.2);
            let params = lorenz_task::DEFAULT;

            lorenz_task::simulate(steps, params, init_values)
        }

        fn variables() -> usize {
            3
        }
    }

    pub struct SinSeries;
    impl TimeSeries for SinSeries {
        fn generate(steps: usize) -> Array2<f64> {
            let mut s: Array2<f64> = Array::zeros((T as usize, 1));

            for (i, mut x) in s.iter_mut().enumerate() {
                *x = sin_t(i as u32);
            }

            s
        }
        fn variables() -> usize {
            1
        }
    }

    /// SinSeries, but with randomly assigned frequencies
    pub struct RandomSinSeries;
    impl TimeSeries for RandomSinSeries {
        fn generate(steps: usize) -> Array2<f64> {
            let mut s: Array2<f64> = Array::zeros((T as usize, 1));

            let a = 2.0;
            let b = rand::thread_rng().gen_range(0.03..0.1);
            let c = 4.0;

            for (i, mut x) in s.iter_mut().enumerate() {
                *x = sin_t_p(i as u32, a, b, c);
            }

            s
        }

        fn variables() -> usize {
            1
        }
    }
}

fn sin_t(t: u32) -> f64 {
    math::ml::sigmoid(2.0 * f32::sin(0.04 * t as f32) - 4.0).into()
}

fn sin_t_p(t: u32, a: f32, b: f32, c: f32) -> f64 {
    math::ml::sigmoid(a * f32::sin(b * t as f32) - c).into()
}
