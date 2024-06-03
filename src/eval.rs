use crate::models::Model;
use crate::runnable::RunnableNetwork;

use tasks::{Task, TaskEval};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::{Evaluate, EvolutionEnvironment};

use model::neuron::izhikevich::Izhikevich;
use model::network::representation::{NetworkRepresentation, NeuronDescription};
use model::network::builder::NetworkBuilder;

use utils::config::ConfigSection;


pub struct EvalConfig {
    pub batch_size: usize,
    pub batch_index: usize,
    //pub n_trials: usize
}

pub struct ModelEvaluator<T: Task + TaskEval> {
    setups: Vec<T::Setup>,
    config: Option<EvalConfig>
}

impl<M, T> Evaluate<M, NetworkRepresentation<NeuronDescription<Izhikevich>>> for ModelEvaluator<T>
where
    T: Task + TaskEval,
    M: Model,
{
    fn eval(&self, m: &M) -> (f32, NetworkRepresentation<NeuronDescription<Izhikevich>>) {
        let repr = m.develop();

        let setup: &[T::Setup];

        if let Some(conf) = &self.config {
            setup = &self.setups[conf.batch_index..(conf.batch_index + conf.batch_size)];
        } else {
            setup = &self.setups;
        }

        (evaluate_network_representation::<T>(&repr, setup), repr)
    }
    fn next(&mut self) {
        if let Some(conf) = &mut self.config {
            conf.batch_index = (conf.batch_index + conf.batch_size) % self.setups.len();
            log::trace!("using next batch with index {}", conf.batch_index);
        }
    }
}

impl<T: Task + TaskEval> ModelEvaluator<T> {
    pub fn new(config: Option<EvalConfig>) -> ModelEvaluator<T> {
        ModelEvaluator {
            config,
            setups: T::eval_setups(),
        }
    }
}

fn evaluate_network_representation<T: Task + TaskEval> (
    repr: &NetworkRepresentation<NeuronDescription<Izhikevich>>,
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

    log::trace!("eval::evaluate_on_task");

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
