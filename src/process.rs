pub mod default;
pub mod hyper;

use crate::analysis::graph::{Graph, GraphAnalysis};
use crate::eval::MultiEvaluator;
use crate::eval::config::{Batch, BatchConfig, EvalConfig};
use crate::config::{get_config, BaseConfig};
use crate::optimization::{Optimizer, OptimizationConfig};

use crate::models::rsnn::RSNNModel;
use crate::models::srsnn::er_model::ERModel;
use crate::models::srsnn::typed::TypedModel;
use crate::models::plain::PlainModel;

use model::Model;
use model::network::representation::DefaultRepresentation;

use tasks::{Task, TaskEval};
use tasks::pattern_task::PatternTask;
use tasks::catching_task::CatchingTask;
use tasks::xor_task::XORTask;
use tasks::pole_balancing_task::PoleBalancingTask;
use tasks::mnist_task::MNISTTask;
//use tasks::movement_task::MovementTask;
//use tasks::survival_task::SurvivalTask;
//use tasks::energy_task::EnergyTask;

use evolution::algorithm::Algorithm;

use serde::Serialize;

use utils::environment::Environment;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


pub trait Process: Sync {
    //type Output;

    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig);

    fn init(config: BaseConfig) {
        Self::resolve_m(config);
    }

    fn resolve_m(config: BaseConfig) {
        match config.model.as_str() {
            "plain"         => { Self::resolve_t::<RSNNModel<PlainModel>>(config); },
            "er_model"      => { Self::resolve_t::<RSNNModel<ERModel>>(config); },
            "typed"   => { Self::resolve_t::<RSNNModel<TypedModel>>(config); },
            //"main" => { Self::resolve_t::<MainStochasticModel>(config); },
            //"matrix" => { Self::resolve_t::<MatrixModel>(config); },
            _ => { println!("Unknown model: {}", config.model); }
        }
    }

    fn resolve_t<M: Model>(config: BaseConfig) {
        match config.task.as_str() {
            "polebalance" => { Self::run::<M, PoleBalancingTask>(config); },
            "pattern"     => { Self::run::<M, PatternTask>(config); },
            "catching"    => { Self::run::<M, CatchingTask>(config); },
            "xor"         => { Self::run::<M, XORTask>(config); },
            "mnist"       => { Self::run::<M, MNISTTask>(config); },
            //"movement"    => { Self::run::<M, MovementTask>(config); },
            //"survival"    => { Self::run::<M, SurvivalTask>(config); },
            //"energy"      => { Self::run::<M, EnergyTask>(config); },
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

    fn evaluator<T: Task + TaskEval>(base_conf: &BaseConfig, eval_conf: &EvalConfig) -> MultiEvaluator<T> {
        let batch_conf = match base_conf.task.as_str() {
            "mnist" | "pattern" => {
                let bc = get_config::<Batch>();

                log::info!("Batch config:\n{:#?}", bc);

                Some(BatchConfig {batch_size: bc.batch_size})
            },
             _ => None
        };

        MultiEvaluator::new(eval_conf.clone(), batch_conf)
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
        log::info!("\n[Configs] \n\
                model = {:#?}\n\
                algorithm = {:#?}\n\
                eval = {:#?}\n\
                optimizer= {:#?}",
                main_config.model, main_config.algorithm, main_config.eval,
                main_config.optimizer);
    }

    fn save<S: Serialize>(object: S, name: String) {
        let filename = format!("out/{}.json", name);

        let r = utils::data::save::<S>(object, filename.as_str());

        match r {
            Ok(_) => {log::info!("Saved data to {}", filename);},
            Err(e) => { log::error!("Could not save model: {e}"); }
        }
    }

    fn analyze_network(r: &DefaultRepresentation) {
        log::debug!("Analyzing network..");

        let g: Graph = r.into();
        let ga = GraphAnalysis::analyze(&g);

        log::info!("Graph: {}\n{}", g, ga);

        let n_inhibitory: f32 = r.neurons.iter()
            .map(|n| f32::from(n.inhibitory)).sum();

        println!("\n    inhibitory: {} ({:.3}%)", n_inhibitory, (n_inhibitory / r.neurons.len() as f32) * 100.0);

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
