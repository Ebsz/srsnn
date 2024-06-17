pub mod config;
pub mod trial;

use crate::eval::trial::{SingleTrialEvaluator, MultiTrialEvaluator};
use crate::eval::config::BatchConfig;

use crate::models::Model;
use crate::runnable::RunnableNetwork;

use tasks::{Task, TaskEval};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::Evaluate;

use model::network::representation::DefaultRepresentation;
use model::network::builder::NetworkBuilder;


struct BatchSetup {
    pub batch_size: usize,
    pub batch_index: usize,
}

/// Performs the actual evaluating
pub trait BaseEvaluator<M, T: Task + TaskEval> {
    fn evaluate(&self, m: &M, setups: &[T::Setup]) -> (f32, DefaultRepresentation);
}

pub struct Evaluator<T: Task + TaskEval> {
    main: MainEvaluator,

    setups: Vec<T::Setup>,
    batch_setup: Option<BatchSetup>
}

impl<M: Model, T: Task + TaskEval> Evaluate<M, DefaultRepresentation> for Evaluator<T> {
    fn eval(&self, m: &M) -> (f32, DefaultRepresentation) {
        self.main.evaluate::<M, T>(m, self.eval_setup())
    }

    fn next(&mut self) {
        if let Some(bs) = &mut self.batch_setup {
            bs.batch_index = (bs.batch_index + bs.batch_size) % self.setups.len();
            log::trace!("using next batch with index {}", bs.batch_index);
        }
    }
}

impl<T: Task + TaskEval> Evaluator<T> {
    pub fn new(batch_config: Option<BatchConfig>, main: MainEvaluator) -> Evaluator<T> {
        let batch_setup = match batch_config {
            Some(conf) => Some(BatchSetup {
                batch_size: conf.batch_size,
                batch_index: 0

            }),
            _ => None
        };

        Evaluator {
            main,
            batch_setup,
            setups: T::eval_setups(),
        }
    }

    fn eval_setup(&self) -> &[T::Setup] {
        if let Some(bc) = &self.batch_setup{
            &self.setups[bc.batch_index..(bc.batch_index + bc.batch_size)]
        } else {
            &self.setups
        }
    }
}

pub enum MainEvaluator {
    SingleTrial(SingleTrialEvaluator),
    MultiTrial(MultiTrialEvaluator),
}

impl MainEvaluator {
    fn evaluate<M: Model, T: Task + TaskEval>(&self, m: &M, setups: &[T::Setup]) -> (f32, DefaultRepresentation) {
        match self {
            MainEvaluator::SingleTrial(s) => <SingleTrialEvaluator as BaseEvaluator<M, T>>::evaluate(s, m, setups),
            MainEvaluator::MultiTrial(s) => <MultiTrialEvaluator as BaseEvaluator<M, T>>::evaluate(s, m, setups),
        }
    }
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

    f
}
