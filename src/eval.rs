pub mod config;
pub mod setups;

use crate::eval::config::{BatchConfig, EvalConfig};
use crate::eval::setups::{EvalSetup, BatchSetup};

use model::Model;
use crate::runnable::RunnableNetwork;

use tasks::{Task, TaskEval};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::Evaluate;

use model::DefaultNetwork;
use model::network::representation::DefaultRepresentation;

use utils::math;
use utils::config::Configurable;

use std::thread;
use std::sync::Arc;
use std::time::Instant;

use crossbeam::queue::ArrayQueue;


pub type Trial = (u32, DefaultRepresentation);
pub type Evaluation = (u32, f32, DefaultRepresentation);

/// Evaluates a network on one or more setups and returns the evaluation over them.
pub fn evaluate_on_task<T: Task + TaskEval> (
    repr: &DefaultRepresentation,
    setups: &[T::Setup]
) -> f32 {

    let mut results: Vec<T::Result> = Vec::new();
    let mut r = RunnableNetwork::<DefaultNetwork>::build(repr);

    for s in setups {
        let task = T::new(s);

        let mut runner = TaskRunner::new(task, &mut r);
        let result = runner.run();

        results.push(result);

        r.reset();
    }

    let f = T::fitness(results);

    f
}

#[derive(Clone)]
pub struct MultiEvaluator<T: Task + TaskEval> {
    setup: EvalSetup<T>,
    config: EvalConfig,
}

impl<M: Model, T: Task + TaskEval> Evaluate<M, DefaultRepresentation> for MultiEvaluator<T> {
    fn eval(&mut self, models: &[(u32, &M)]) -> Vec<Evaluation> {
        let t0 = Instant::now();

        let n_samples = models.len() * self.config.trials;

        let input_queue: Arc<ArrayQueue<Trial>> = Arc::new(ArrayQueue::new(n_samples));
        let output_queue: Arc<ArrayQueue<Evaluation>> = Arc::new(ArrayQueue::new(n_samples));


        log::trace!("Developing models");
        for m in models {
            for _ in 0..self.config.trials {
                let _ = input_queue.push((m.0, m.1.develop()));
            }
        }
        log::trace!("Developed population in {:.2}s", t0.elapsed().as_secs_f32());

        assert!(input_queue.len() == n_samples);

        let setup = (*(self.setup.get())).to_vec();

        // Don't create more threads than there are objects to evaluate
        let n_threads = std::cmp::min(self.config.max_threads, models.len());

        log::trace!("Starting eval");
        thread::scope(|s| {
            for _ in 0..n_threads {
                let iq = input_queue.clone();
                let oq = output_queue.clone();

                let sref = &setup[..];

                s.spawn(move || {
                    while let Some(t) = iq.pop() {
                        let eval = evaluate_on_task::<T>(&t.1, sref);

                        let _ = oq.push((t.0, eval, t.1));
                    }
                });
            }
        });

        let mut evals = vec![];
        while let Some(e) = output_queue.pop() {
            evals.push(e);
        }

        // handle multitrial evals
        if self.config.trials > 1 {
            evals = self.multitrial_evals(evals, models.len());
        }

        self.setup.next();

        assert!(evals.len() == models.len());

        let elapsed_t = t0.elapsed().as_secs_f32();

        log::debug!("Finished {} evals in {:.3}s ({:.2} evals per second)",
            evals.len(),  elapsed_t, evals.len() as f32 / elapsed_t);

        evals
    }
}

impl<T: Task + TaskEval> MultiEvaluator<T> {
    pub fn new(config: EvalConfig, batch_config: Option<BatchConfig>) -> MultiEvaluator<T> {
        let setup = match batch_config {
            Some(bc) => EvalSetup::Batched(BatchSetup::new(T::eval_setups(), bc.batch_size)),
            None => EvalSetup::Base(T::eval_setups())
        };

        MultiEvaluator {
            setup,
            config
        }
    }

    pub fn multitrial_evals(&self, mut e: Vec<Evaluation>, n_models: usize) -> Vec<Evaluation> {
        e.sort_by_key(|x| x.0);

        let mut evals = vec![];
        for i in 0..n_models {
            let ix = i * self.config.trials;
            let model_evals = &e[ix..ix+self.config.trials];

            // 1. find the best one to return
            let best_index = math::max_index(model_evals.iter().map(|x| x.1));

            // 2. calculate average fitness
            let avg_eval: f32 = model_evals.iter().map(|x| x.1).sum::<f32>() /  self.config.trials as f32;

            evals.push((model_evals[0].0, avg_eval, model_evals[best_index].2.clone()));
        }

        evals
    }
}

impl<T: Task + TaskEval> Configurable for MultiEvaluator<T> {
    type Config = EvalConfig;
}

use tasks::pattern_task::{PatternTask, PatternTaskResult, validation_setups};
use ndarray::array;

pub fn validate_pattern_task(r: &DefaultRepresentation) -> f32 {
    const N_VALIDATIONS: usize = 100;

    let setups = validation_setups(N_VALIDATIONS);

    let mut results = vec![];

    for s in &setups {
        let task = PatternTask::new(s);
        let mut runnable = RunnableNetwork::<DefaultNetwork>::build(r);

        runnable.network.enable_recording();

        let mut runner = TaskRunner::new(task, &mut runnable);
        let res = runner.run();

        results.push(res);
    }

    let acc = calculate_accuracy(&results);
    log::info!("accuracy: {acc}");

    PatternTask::fitness(results)
}

fn calculate_accuracy(results: &Vec<PatternTaskResult>) -> f32 {
    let mut correct = 0;

    for r in results {
        let (w,h) = (r.output.shape()[0], r.output.shape()[1]);

        let avg_fr = r.output.iter().sum::<u32>() as f32 /  (w * h)  as f32;

        // p[0]: patterns are equal, p[1]: patterns not equal
        let prediction = array![1.0 - avg_fr, avg_fr];

        if r.is_same && prediction[0] > prediction[1]  {
            correct += 1;
        } else if !r.is_same && prediction[1] > prediction[0] {
            correct += 1;
        }
    }

    correct as f32 / results.len() as f32
}
