use crate::models::Model;
use crate::runnable::RunnableNetwork;

use tasks::{Task, TaskEval};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::{Evaluate, EvolutionEnvironment};

use model::neuron::izhikevich::Izhikevich;
use model::network::Network;
use model::network::description::{NetworkDescription, NeuronDescription};
use model::network::builder::NetworkBuilder;


pub struct ModelEvaluator<T: Task + TaskEval> {
    main_evaluator: TaskEvaluator<T>,
    env: EvolutionEnvironment
}

impl<M, T> Evaluate<M, NetworkDescription<NeuronDescription<Izhikevich>>> for ModelEvaluator<T>
where
    T: Task + TaskEval,
    M: Model,
{
    fn eval(&self, m: &M) -> (f32, NetworkDescription<NeuronDescription<Izhikevich>>) {
        let desc = m.develop();

        let network = NetworkBuilder::build(&desc);

        let mut runnable = RunnableNetwork {
            network,
            inputs: self.env.inputs,
            outputs: self.env.outputs
        };

        (self.main_evaluator.evaluate_on_task(&mut runnable), desc)
    }
}

impl<T: Task + TaskEval> ModelEvaluator<T> {
    pub fn new(env: EvolutionEnvironment) -> ModelEvaluator<T> {
        ModelEvaluator {
            main_evaluator: TaskEvaluator::new(),
            env
        }
    }
}

pub struct TaskEvaluator<T: Task + TaskEval> {
    setups: Vec<T::Setup>,
}

impl<T: Task + TaskEval> TaskEvaluator<T> {
    pub fn new() -> Self {
        log::trace!("Creating evaluator");
        TaskEvaluator {
            setups: T::eval_setups(),
        }
    }

    pub fn evaluate_on_task<R: Runnable>(&self, r: &mut R) -> f32 {
        let mut results: Vec<T::Result> = Vec::new();

        for s in &self.setups {
            let task = T::new(s);

            let mut runner = TaskRunner::new(task, r);
            let result = runner.run();

            results.push(result);
        }

        let f = T::fitness(results);
        log::trace!("eval: {}", f);

        f
    }
}
