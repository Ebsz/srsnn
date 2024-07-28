use crate::eval::MultiEvaluator;
use crate::eval::config::{Batch, BatchConfig};
use crate::config::{get_config, BaseConfig};

use crate::models::rsnn::RSNNModel;
use crate::models::srsnn::er_model::ERModel;
use crate::models::srsnn::typed::TypedModel;
use crate::models::plain::PlainModel;


use model::Model;

use tasks::{Task, TaskEval};
use tasks::mnist_task::MNISTTask;
use tasks::catching_task::CatchingTask;
use tasks::movement_task::MovementTask;
use tasks::survival_task::SurvivalTask;
use tasks::energy_task::EnergyTask;
use tasks::xor_task::XORTask;
use tasks::pole_balancing_task::PoleBalancingTask;

use utils::environment::Environment;


pub trait Process {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig);

    fn init(config: BaseConfig) {
        Self::resolve_m(config);
    }

    fn resolve_m(config: BaseConfig) {
        match config.model.as_str() {
            "er_model" => { Self::resolve_t::<RSNNModel<ERModel>>(config); },
            "typed_model" => { Self::resolve_t::<RSNNModel<TypedModel>>(config); },
            "plain" => { Self::resolve_t::<RSNNModel<PlainModel>>(config); },
            //"main" => { Self::resolve_t::<MainStochasticModel>(config); },
            //"matrix" => { Self::resolve_t::<MatrixModel>(config); },
            //"rsnn" => { Self::resolve_t::<RSNNModel>(config); },
            _ => { println!("Unknown model: {}", config.model); }
        }
    }

    fn resolve_t<M: Model>(config: BaseConfig) {
        match config.task.as_str() {
            "polebalance" => { Self::run::<M, PoleBalancingTask>(config); },
            "catching"    => { Self::run::<M, CatchingTask>(config); },
            "movement"    => { Self::run::<M, MovementTask>(config); },
            "survival"    => { Self::run::<M, SurvivalTask>(config); },
            "energy"      => { Self::run::<M, EnergyTask>(config); },
            "mnist"       => { Self::run::<M, MNISTTask>(config); },
            "xor"         => { Self::run::<M, XORTask>(config); },
            _ => { println!("Unknown task: {}", config.task); }
        }
    }

    fn environment<T: Task>() -> Environment {
        let e = T::environment();

        Environment {
            inputs: e.agent_inputs,
            outputs: e.agent_outputs,
        }
    }

    fn evaluator<T: Task + TaskEval>(config: &BaseConfig) -> MultiEvaluator<T> {
        let batch_config = match config.task.as_str() {
            "mnist" => {
                let bc = get_config::<Batch>();

                log::info!("Batch config:\n{:#?}", bc);

                Some(BatchConfig {batch_size: bc.batch_size})
            },
             _ => None
        };

        let config = get_config::<MultiEvaluator<T>>();
        log::info!("Eval config:\n{:#?}", config);

        MultiEvaluator::new(config, batch_config)
    }
}
