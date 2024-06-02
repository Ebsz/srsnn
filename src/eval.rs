use crate::models::Model;
use crate::runnable::RunnableNetwork;

use tasks::{Task, TaskEval};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::{Evaluate, EvolutionEnvironment};

use model::neuron::izhikevich::Izhikevich;
use model::network::representation::{NetworkRepresentation, NeuronDescription};
use model::network::builder::NetworkBuilder;


pub struct ModelEvaluator<T: Task + TaskEval> {
    env: EvolutionEnvironment,
    setups: Vec<T::Setup>
}

impl<M, T> Evaluate<M, NetworkRepresentation<NeuronDescription<Izhikevich>>> for ModelEvaluator<T>
where
    T: Task + TaskEval,
    M: Model,
{
    fn eval(&self, m: &M) -> (f32, NetworkRepresentation<NeuronDescription<Izhikevich>>) {
        let desc = m.develop();

        let network = NetworkBuilder::build(&desc);

        let mut runnable = RunnableNetwork {
            network,
            inputs: self.env.inputs,
            outputs: self.env.outputs,
        };

        (evaluate_on_task::<T, _>(&mut runnable, &self.setups), desc)
    }
}

impl<T: Task + TaskEval> ModelEvaluator<T> {
    pub fn new(env: EvolutionEnvironment) -> ModelEvaluator<T> {
        ModelEvaluator {
            env,
            setups: T::eval_setups(),
        }
    }
}

fn evaluate_on_task<T: Task + TaskEval, R: Runnable> (
    r: &mut R,
    setups: &Vec<T::Setup>
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
