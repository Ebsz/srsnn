use tasks::{Task, TaskEval};
use tasks::task_runner::TaskRunner;

use evolution::{Evaluate, EvolutionEnvironment};
use crate::phenotype::EvolvableGenome;


pub struct TaskEvaluator<T: Task + TaskEval, G: EvolvableGenome> {
    setups: Vec<T::Setup>,
    genome: Option<G>, // NOTE: not used, required..
    env: EvolutionEnvironment
}

impl<T: Task + TaskEval, G: EvolvableGenome> Evaluate<G> for TaskEvaluator<T, G> {
    fn eval(&self, g: &G) -> f32 {
        let mut results: Vec<T::Result> = Vec::new();

        for s in &self.setups {
            let task = T::new(s);
            let mut phenotype = g.to_phenotype(&self.env);
            let mut runner = TaskRunner::new(task, &mut phenotype);
            let result = runner.run();

            results.push(result);
        }

        let f = T::fitness(results);
        log::trace!("eval: {}", f);

        f
    }
}

impl<T: Task + TaskEval, G: EvolvableGenome> TaskEvaluator<T,  G> {
    pub fn new(env: EvolutionEnvironment) -> Self {
        log::trace!("Creating evaluator");
        TaskEvaluator {
            setups: T::eval_setups(),
            genome: None,
            env
        }
    }
}
