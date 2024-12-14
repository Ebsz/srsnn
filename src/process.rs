pub mod default;
pub mod experiment;
pub mod test;
pub mod hyper;

use crate::eval::MultiEvaluator;
use crate::eval::config::{Batch, BatchConfig, EvalConfig};
use crate::config::{get_config, BaseConfig};
use crate::optimization::{Optimizer, OptimizationConfig};

use crate::models::generator_model::GeneratorModel;

use crate::models::generator::uniform::UniformModel;
use crate::models::generator::typed_uniform::TypedUniformModel;
use crate::models::generator::base::BaseModel;
use crate::models::generator::ed::EvolvedDynamicsModel;
use crate::models::generator::er::ER0Model;

use crate::models::generator::udd_base::UDDBaseModel;
use crate::models::generator::utd_base::UTDBaseModel;
use crate::models::generator::edd_base::EDDModel;

use crate::models::generator::ex1_ablation::GeometricModel;
use crate::models::generator::ex1_ablation::TypedModel;
use crate::models::generator::ex1_ablation::GeometricTypedModel;

use model::Model;

use tasks::{Task, TaskEval};
use tasks::catching_task::CatchingTask;
use tasks::xor_task::XORTask;
use tasks::pattern::PatternTask;
use tasks::mnist_task::MNISTTask;
use tasks::multipattern::MultiPatternTask;
use tasks::pattern_similarity::PatternSimilarityTask;

use tasks::time_series::TimeSeriesTask;
use tasks::time_series::ts::{SinSeries, RandomSinSeries};

use tasks::testing::TestTask;

use evolution::algorithm::Algorithm;

use serde::Serialize;

use utils::environment::Environment;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


pub trait Process: Sync {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig);

    fn init(config: BaseConfig) {
        Self::resolve_m(config);
    }

    fn resolve_m(config: BaseConfig) {
        match config.model.as_str() {
            "base_model"                => { Self::resolve_t::<GeneratorModel<BaseModel>>(config); },
            "ed_model"                  => { Self::resolve_t::<GeneratorModel<EvolvedDynamicsModel>>(config); },
            "uniform_model"             => { Self::resolve_t::<GeneratorModel<UniformModel>>(config); },
            "er0_model"                 => { Self::resolve_t::<GeneratorModel<ER0Model>>(config); },
            "typed_uniform_model"       => { Self::resolve_t::<GeneratorModel<TypedUniformModel>>(config); },
            "udd_base_model"            => { Self::resolve_t::<GeneratorModel<UDDBaseModel>>(config); },
            "utd_base_model"            => { Self::resolve_t::<GeneratorModel<UTDBaseModel>>(config); },
            "edd_base_model"            => { Self::resolve_t::<GeneratorModel<EDDModel>>(config); },
            "geometric_uniform"         => { Self::resolve_t::<GeneratorModel<GeometricModel>>(config); },
            "typed_model"               => { Self::resolve_t::<GeneratorModel<TypedModel>>(config); },
            "geometric_typed_model"     => { Self::resolve_t::<GeneratorModel<GeometricTypedModel>>(config); },
            _ => { println!("Unknown model: {}", config.model); }
        }
    }

    /// NOTE: When adding a new task, to enable batching, it must also be added to the
    /// condition in the evaluator() function below. It's stupid, but I don't have time to fix it right now.
    fn resolve_t<M: Model>(config: BaseConfig) {
        match config.task.as_str() {
            "pattern"               => { Self::run::<M, PatternTask>(config); },
            "pattern_similarity"    => { Self::run::<M, PatternSimilarityTask>(config); },
            "multipattern"          => { Self::run::<M, MultiPatternTask>(config); },
            "catching"              => { Self::run::<M, CatchingTask>(config); },
            "xor"                   => { Self::run::<M, XORTask>(config); },
            "mnist"                 => { Self::run::<M, MNISTTask>(config); },
            "testing"               => { Self::run::<M, TestTask>(config); },
            "single_sin_time_series"       => { Self::run::<M, TimeSeriesTask<SinSeries>>(config); },
            "multi_sin_time_series"       => { Self::run::<M, TimeSeriesTask<RandomSinSeries>>(config); },
            _ => { println!("Unknown task: {}", config.task); }
        }
    }

    fn main_conf<M: Model, T: Task + TaskEval, A: Algorithm>() -> MainConf<M, A> {
        MainConf {
            model: get_config::<M>(),
            algorithm: get_config::<A>(),
            eval: get_config::<MultiEvaluator<T>>(),
            optimizer: get_config::<Optimizer>(),
        }
    }

    fn environment<T: Task>() -> Environment {
        let e = T::environment();

        Environment {
            inputs: e.agent_inputs,
            outputs: e.agent_outputs,
        }
    }

    fn evaluator<T: Task + TaskEval>(base_conf: &BaseConfig, eval_conf: &EvalConfig, setups: Vec<T::Setup>) -> MultiEvaluator<T> {
        let batch_conf = match base_conf.task.as_str() {
            "mnist" | "pattern" | "multipattern" | "pattern_similarity" => {
                let bc = get_config::<Batch>();

                log::info!("Batch config:\n{:#?}", bc);

                Some(BatchConfig {batch_size: bc.batch_size})
            },
             _ => None
        };

        MultiEvaluator::new(eval_conf.clone(), batch_conf, setups)
    }

    fn init_ctrl_c_handler(stop_signal: Arc<AtomicBool>) {
        let mut stopped = false;

        ctrlc::set_handler(move || {
            if stopped {
                std::process::exit(1);
            } else {
                log::info!("Stopping..");

                stopped = true;
                stop_signal.store(true, Ordering::SeqCst);
            }
        }).expect("Error setting Ctrl-C handler");

        log::info!("Use Ctrl-C to stop gracefully");
    }

    fn log_config<M: Model, A: Algorithm>(
        base_config: &BaseConfig,
        main_config: &MainConf<M, A>,
        env: &Environment) {
        log::info!("Model: {} ({} params)", base_config.model, M::params(&main_config.model, env).size());
        log::info!("Task: {}", base_config.task);
        log::info!("Configs: \n\
                model = {:#?}\n\
                algorithm = {:#?}\n\
                eval = {:#?}\n\
                optimizer = {:#?}\n\
                environment = {:#?}",
                main_config.model, main_config.algorithm, main_config.eval,
                main_config.optimizer, env);
    }

    fn save<S: Serialize>(object: S, name: String) {
        let filename = format!("out/{}.json", name);

        let r = utils::data::save::<S>(object, filename.as_str());

        match r {
            Ok(_) => {log::info!("Saved data to {}", filename);},
            Err(e) => { log::error!("Could not save model: {e}"); }
        }
    }
}

// TODO: Rename to something else
#[derive(Debug, Clone)]
pub struct MainConf<M: Model, A: Algorithm> {
    pub model: M::Config,
    pub algorithm: A::Config,
    pub eval: EvalConfig,
    pub optimizer: OptimizationConfig
}
