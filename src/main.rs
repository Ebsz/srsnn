use luna::eval::MultiEvaluator;
use luna::config::{base_config, BaseConfig};
use luna::process::{Process, MainConf};
use luna::optimization::Optimizer;
use luna::plots;

use luna::experiment;

use model::Model;
use model::network::representation::DefaultRepresentation;

use tasks::{Task, TaskEval};

use evolution::algorithm::Algorithm;
use evolution::algorithm::nes::NES;
use evolution::stats::EvolutionStatistics;

use utils::random;
use utils::logger::init_logger;

use std::env;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;


fn save_network(network: DefaultRepresentation, config: &BaseConfig) {
    let filename = format!("out/evolved_{}_{}_{}.json",
        config.model, config.task, random::random_range((0,1000000)).to_string());

    let r = utils::data::save::<DefaultRepresentation>(network, filename.as_str());

    match r {
        Ok(_) => {log::info!("Model saved to {}", filename);},
        Err(e) => { log::error!("Could not save model: {e}"); }
    }
}

fn log_config<M: Model, A: Algorithm>(base_config: &BaseConfig, main_config: &MainConf<M, A>) {
    log::info!("Model: {}", base_config.model);
    log::info!("Task: {}", base_config.task);
    log::info!("\n[Configs] \n\
            model = {:#?}\n\
            algorithm = {:#?}\n\
            eval = {:#?}",
            main_config.model, main_config.algorithm, main_config.eval);
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
        let main_conf = Self::main_conf::<M, T, NES>();

        log_config(&conf, &main_conf);

        let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval);
        let env = Self::environment::<T>();

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        let mut stats = Optimizer::optimize::<M, T, NES>(evaluator, &main_conf, env, stop_signal);
        report(&mut stats);
    }
}

fn report(stats: &mut EvolutionStatistics) {
    plots::plot_evolution_stats_all(stats);
}

fn main() {
    let config_name = parse_config_name_from_args();
    let config = base_config(config_name.clone());

    init_logger(config.log_level.clone());
    log::debug!("Using config: {}", config_name.unwrap_or("default".to_string()));
    log::debug!("seed is {}", random::SEED);

    //OptimizationProcess::init(config);

    experiment::HyperOptimization::init(config);
}
