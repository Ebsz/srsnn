use crate::eval::{evaluate_network_representation, BaseEvaluator};
use crate::eval::config::TrialConfig;
use crate::models::Model;

use tasks::{Task, TaskEval};

use model::network::representation::DefaultRepresentation;

use utils::config::Configurable;


pub struct SingleTrialEvaluator;

impl<M: Model, T: Task + TaskEval> BaseEvaluator<M, T> for SingleTrialEvaluator {
    fn evaluate(&self, m: &M, setups: &[T::Setup]) -> (f32, DefaultRepresentation) {
        let repr = m.develop();

        let eval = evaluate_network_representation::<T>(&repr, setups);

        (eval, repr)
    }
}

pub struct MultiTrialEvaluator {
    pub config: TrialConfig
}

impl<M: Model, T: Task + TaskEval> BaseEvaluator<M, T> for MultiTrialEvaluator {
    fn evaluate(&self, m: &M, setups: &[T::Setup]) -> (f32, DefaultRepresentation) {
        let mut evals: Vec<(f32, DefaultRepresentation)> = Vec::new();

        for _ in 0..self.config.trials {
            let repr = m.develop();
            let eval = evaluate_network_representation::<T>(&repr, setups);
            evals.push((eval, repr));
        }

        evals.sort_by(|x,y| y.0.partial_cmp(&x.0).unwrap());

        let avg_eval = evals.iter().map(|(e, _)| e).sum::<f32>() / self.config.trials as f32;
        let best_eval: (f32, DefaultRepresentation) = evals.remove(0);

        log::info!("Average eval: {avg_eval}, best: {}", best_eval.0);

        (avg_eval, best_eval.1)
    }
}

impl Configurable for MultiTrialEvaluator {
    type Config = TrialConfig;
}
