pub mod config;
pub mod setups;

use crate::eval::config::{BatchConfig, EvalConfig};
use crate::eval::setups::{EvalSetup, BatchSetup};

use crate::models::Model;
use crate::runnable::RunnableNetwork;

use tasks::{Task, TaskEval};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::Evaluate;

use model::DefaultNetwork;
use model::network::SpikingNetwork;
use model::network::representation::DefaultRepresentation;
use model::network::builder::NetworkBuilder;

use utils::math;
use utils::config::Configurable;

use std::thread;
use std::sync::{Arc, Mutex};

use crossbeam::queue::ArrayQueue;


/// Evaluate a network on a number of different setups, returning the evaluation over them.
pub fn evaluate_on_task<T: Task + TaskEval> (
    repr: &DefaultRepresentation,
    setups: &[T::Setup]
) -> f32 {
    let mut r = RunnableNetwork::<DefaultNetwork>::build(repr);

    let mut results: Vec<T::Result> = Vec::new();

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


type Trial = (u32, DefaultRepresentation);
type Evaluation = (u32, f32, DefaultRepresentation);

pub struct MultiEvaluator<T: Task + TaskEval> {
    setup: EvalSetup<T>,
    config: EvalConfig,
}

impl<M: Model, T: Task + TaskEval> Evaluate<M, DefaultRepresentation> for MultiEvaluator<T> {
    fn eval(&mut self, genomes: &[(u32, &M)]) -> Vec<Evaluation> {
        let n_samples = genomes.len() * self.config.trials;

        let input_queue: Arc<ArrayQueue<Trial>> = Arc::new(ArrayQueue::new(n_samples));
        let output_queue: Arc<ArrayQueue<(u32, f32, DefaultRepresentation)>> = Arc::new(ArrayQueue::new(n_samples));

        for g in genomes {
            for _ in 0..self.config.trials {
                input_queue.push((g.0, g.1.develop()));
            }
        }

        assert!(input_queue.len() == n_samples);

        let setup = (*(self.setup.get()).clone()).to_vec();

        let mut handles: Vec<thread::JoinHandle<()>> = vec![];

        let n_threads = std::cmp::min(self.config.max_threads, genomes.len());
        thread::scope(|s| {
            for _ in 0..n_threads {
                let iq = input_queue.clone();
                let oq = output_queue.clone();

                let sref = &setup[..];

                s.spawn(move || {
                    while let Some(t) = iq.pop() {
                        let eval = evaluate_on_task::<T>(&t.1, sref);

                        oq.push((t.0, eval, t.1));
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
            evals = self.multitrial_evals(evals, genomes.len());
        }

        self.setup.next();

        assert!(evals.len() == genomes.len());
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

    pub fn multitrial_evals(&self, mut e: Vec<Evaluation>, genomes: usize) -> Vec<Evaluation> {
        e.sort_by_key(|x| x.0);

        let mut evals = vec![];
        for i in 0..genomes {
            let ix = i * self.config.trials;
            let genome_evals = &e[ix..ix+self.config.trials];

            // 1. find the best one to return
            let best_index = math::max_index(genome_evals.iter().map(|x| x.1));

            // 2. calculate average fitness
            let avg_eval: f32 = genome_evals.iter().map(|x| x.1).sum::<f32>() /  self.config.trials as f32;

            evals.push((genome_evals[0].0, avg_eval, genome_evals[best_index].2.clone()));

        }

        evals
    }
}

impl<T: Task + TaskEval> Configurable for MultiEvaluator<T> {
    type Config = EvalConfig;
}

