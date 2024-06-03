use crate::models::Model;
use crate::runnable::RunnableNetwork;

use tasks::{Task, TaskEval};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::{Evaluate, EvolutionEnvironment};

use model::neuron::izhikevich::Izhikevich;
use model::network::representation::DefaultRepresentation;
use model::network::builder::NetworkBuilder;

use utils::config::{Configurable, ConfigSection};

use serde::Deserialize;


pub struct ModelEvaluator<T: Task + TaskEval> {
    setups: Vec<T::Setup>,
    config: EvalConfig
}

impl<M, T> Evaluate<M, DefaultRepresentation> for ModelEvaluator<T>
where
    T: Task + TaskEval,
    M: Model,
{
    fn eval(&self, m: &M) -> (f32, DefaultRepresentation) {
        let setup = self.get_setup();

        let mut evals: Vec<(f32, DefaultRepresentation)> = Vec::new();

        for _ in 0..self.config.trials {
            let repr = m.develop();
            let eval = evaluate_network_representation::<T>(&repr, setup);
            evals.push((eval, repr));
        }

        evals.sort_by(|x,y| y.0.partial_cmp(&x.0).unwrap());

        let avg_eval = evals.iter().map(|(e, _)| e).sum::<f32>() / self.config.trials as f32;
        let best_eval: (f32, DefaultRepresentation) = evals.remove(0);

        log::info!("Average eval: {avg_eval}, best: {}", best_eval.0);

        (avg_eval, best_eval.1)
    }

    fn next(&mut self) {
        if let Some(bc) = &mut self.config.batch {
            bc.batch_index = (bc.batch_index + bc.batch_size) % self.setups.len();
            log::trace!("using next batch with index {}", bc.batch_index);
        }
    }
}

impl<T: Task + TaskEval> ModelEvaluator<T> {
    pub fn new(config: EvalConfig) -> ModelEvaluator<T> {
        ModelEvaluator {
            config,
            setups: T::eval_setups(),
        }
    }

    fn get_setup(&self) -> &[T::Setup] {
        if let Some(bc) = &self.config.batch {
            &self.setups[bc.batch_index..(bc.batch_index + bc.batch_size)]
        } else {
            &self.setups
        }
    }
}

#[derive(Deserialize)]
pub struct EvalConfig {
    pub trials: usize,
    pub batch: Option<BatchConfig>
}

impl ConfigSection for EvalConfig {
    fn name() -> String {
        "eval".to_string()
    }
}

#[derive(Deserialize)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub batch_index: usize,
}

impl<T: Task + TaskEval> Configurable for ModelEvaluator<T> {
    type Config = EvalConfig;
}

fn evaluate_network_representation<T: Task + TaskEval> (
    repr: &DefaultRepresentation,
    setups: &[T::Setup]
) -> f32 {
    let network = NetworkBuilder::build(repr);

    let mut runnable = RunnableNetwork {
        network,
        inputs: repr.inputs,
        outputs: repr.outputs,
    };

    evaluate_on_task::<T, _>(&mut runnable, setups)
}

/// Evaluate a Runnable on a number of different setups, returning the evaluation over them.
fn evaluate_on_task<T: Task + TaskEval, R: Runnable> (
    r: &mut R,
    setups: &[T::Setup]
) -> f32 {
    let mut results: Vec<T::Result> = Vec::new();

    for s in setups {
        let task = T::new(s);

        let mut runner = TaskRunner::new(task, r);
        let result = runner.run();

        results.push(result);

        r.reset();
    }

    let f = T::fitness(results);

    log::debug!("eval: {}", f);

    f
}
