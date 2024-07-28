use luna::eval::MultiEvaluator;
use luna::optimization::{optimize, MainConf};
use luna::config::{get_config, base_config, BaseConfig};
use luna::eval::config::{Batch, BatchConfig};

use luna::models::rsnn::RSNNModel;
use luna::models::srsnn::er_model::ERModel;
use luna::models::srsnn::typed::TypedModel;
use luna::models::plain::PlainModel;

use tasks::{Task, TaskEval};
use tasks::mnist_task::MNISTTask;
use tasks::catching_task::CatchingTask;
use tasks::movement_task::MovementTask;
use tasks::survival_task::SurvivalTask;
use tasks::energy_task::EnergyTask;
use tasks::xor_task::XORTask;
use tasks::pole_balancing_task::PoleBalancingTask;

use model::Model;
use model::network::representation::DefaultRepresentation;

use evolution::algorithm::nes::NES;

use utils::random;
use utils::logger::init_logger;
use utils::environment::Environment;

use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


trait Process {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig);

    fn init(config: BaseConfig) {
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
}

//struct EvolutionProcess;
//
//impl Process for EvolutionProcess {
//    fn run<M: Model + Genome, T: Task + TaskEval>(config: BaseConfig) {
//        log::info!("EvolutionProcess");
//        let env = Self::environment::<T>();
//
//        let genome_config = get_config::<M>();
//        log_config::<M>(&config, &genome_config);
//
//        let evaluator: MultiEvaluator<T> = Self::get_evaluator(&config);
//
//        let mut population = Population::<_, M, DefaultRepresentation>
//            ::new(env.clone(), config.evolution, genome_config, evaluator);
//
//        init_ctrl_c_handler(population.stop_signal.clone());
//
//        let evolved = population.evolve();
//
//        log::info!("best fitness: {:?}", evolved.fitness.unwrap());
//
//        let network = evolved.phenotype.as_ref().unwrap().clone();
//
//        Self::save_evolved_network(network, &config);
//    }
//}
//

fn save_network(network: DefaultRepresentation, config: &BaseConfig) {
    let filename = format!("out/evolved_{}_{}_{}.json",
        config.model, config.task, random::random_range((0,1000000)).to_string());

    let r = utils::data::save::<DefaultRepresentation>(network, filename.as_str());

    match r {
        Ok(_) => {log::info!("Model saved to {}", filename);},
        Err(e) => { log::error!("Could not save model: {e}"); }
    }
}


fn get_evaluator<T: Task + TaskEval>(config: &BaseConfig) -> MultiEvaluator<T> {
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

fn log_config<M: Model>(main_config: &BaseConfig, genome_config: &M::Config) {
    log::info!("Model: {}", main_config.model);
    log::info!("Task: {}", main_config.task);
    //log::info!("evolution config:\n{:#?}", main_config.evolution);
    log::info!("model config:\n{:#?}", genome_config);
}

fn parse_config_name_from_args() -> Option<String> {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        return Some(args[1].clone());
    }

    None
}

struct OptimizationProcess;
impl Process for OptimizationProcess {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        log::info!("OptimizationProcess");

        log::info!("Model: {}", conf.model);
        log::info!("Task: {}", conf.task);

        let main_conf: MainConf::<M, NES> = MainConf {
            model: get_config::<M>(),
            algorithm: get_config::<NES>(),
        };

        let evaluator: MultiEvaluator<T> = get_evaluator(&conf);
        let env = Self::environment::<T>();

        optimize::<M, T, NES>(main_conf, evaluator, env);
    }
}


fn main() {
    let config_name = parse_config_name_from_args();
    let config = base_config(config_name.clone());

    init_logger(config.log_level.clone());
    log::debug!("Using config: {}", config_name.unwrap_or("default".to_string()));
    log::debug!("seed is {}", random::SEED);

    OptimizationProcess::init(config);
}
