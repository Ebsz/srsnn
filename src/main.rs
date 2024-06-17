//! luna/src/main.rs

use luna::config::{get_config, main_config, MainConfig};

use luna::eval::trial::{SingleTrialEvaluator, MultiTrialEvaluator};
use luna::eval::{Evaluator, MainEvaluator};
use luna::eval::config::{Batch, BatchConfig};

use luna::models::Model;
use luna::models::stochastic::main_model::MainStochasticModel;
use luna::models::matrix::MatrixModel;

use model::network::representation::DefaultRepresentation;

use evolution::EvolutionEnvironment;
use evolution::population::Population;

use tasks::{Task, TaskEval};
use tasks::mnist_task::MNISTTask;
use tasks::catching_task::CatchingTask;
use tasks::movement_task::MovementTask;
use tasks::survival_task::SurvivalTask;
use tasks::energy_task::EnergyTask;
use tasks::xor_task::XORTask;
use tasks::pole_balancing_task::PoleBalancingTask;

use utils::logger::init_logger;
use utils::random::SEED;

use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


trait Process {
    fn run<M: Model, T: Task + TaskEval>(conf: MainConfig);

    fn init(config: MainConfig) {
        match config.model.as_str() {
            "main" => { Self::resolve_t::<MainStochasticModel>(config); },
            "matrix" => { Self::resolve_t::<MatrixModel>(config); },
            _ => { println!("Unknown model: {}", config.model); }
        }
    }

    fn resolve_t<M: Model>(config: MainConfig) {
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

    fn environment<T: Task>() -> EvolutionEnvironment {
        let e = T::environment();

        EvolutionEnvironment {
            inputs: e.agent_inputs,
            outputs: e.agent_outputs,
        }
    }
}

struct EvolutionProcess;

impl Process for EvolutionProcess {
    fn run<M: Model, T: Task + TaskEval>(config: MainConfig) {
        log::info!("EvolutionProcess");
        let env = Self::environment::<T>();

        let genome_config = get_config::<M>();
        log_config::<M>(&config, &genome_config);

        let evaluator: Evaluator<T> = Self::get_evaluator(&config);

        let mut population = Population::<_, M, DefaultRepresentation>
            ::new(env.clone(), config.evolution, genome_config, evaluator);

        init_ctrl_c_handler(population.stop_signal.clone());

        let evolved = population.evolve();

        let repr = evolved.phenotype.as_ref().unwrap();

        log::info!("best fitness: {:?}", evolved.fitness.unwrap());
    }
}

impl EvolutionProcess {
    fn get_evaluator<T: Task + TaskEval>(config: &MainConfig) -> Evaluator<T> {
        let base_eval = match config.model.as_str() {
            "main" =>  MainEvaluator::MultiTrial(MultiTrialEvaluator {
                config: get_config::<MultiTrialEvaluator>()
            }),
            _ => MainEvaluator::SingleTrial(SingleTrialEvaluator),
        };

        let batch_config = match config.task.as_str() {
            "mnist" => {
                let bc = get_config::<Batch>();

                log::info!("Batch config:\n{:#?}", bc);

                Some(BatchConfig {batch_size: bc.batch_size})
            },
             _ => None
        };

        Evaluator::new(batch_config, base_eval)
    }
}

fn log_config<M: Model>(main_config: &MainConfig, genome_config: &M::Config) {
    log::info!("Model: {}", main_config.model);
    log::info!("Task: {}", main_config.task);
    log::info!("evolution config:\n{:#?}", main_config.evolution);
    log::info!("model config:\n{:#?}", genome_config);
}

fn init_ctrl_c_handler(stop_signal: Arc<AtomicBool>) {
    let mut stopped = false;

    ctrlc::set_handler(move || {
        if stopped {
            std::process::exit(1);
        } else {
            log::info!("Stopping evolution..");

            stopped = true;
            stop_signal.store(true, Ordering::SeqCst);
        }
    }).expect("Error setting Ctrl-C handler");

    log::info!("Use Ctrl-C to stop evolution");
}

fn parse_config_name_from_args() -> Option<String> {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        return Some(args[1].clone());
    }

    None
}

fn main() {
    let config_name = parse_config_name_from_args();
    let config = main_config(config_name.clone());

    init_logger(config.log_level.clone());

    log::info!("Using config: {}", config_name.unwrap_or("default".to_string()));
    log::debug!("seed is {}", SEED);

    EvolutionProcess::init(config);
}
