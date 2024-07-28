use luna::eval::MultiEvaluator;
use luna::eval::config::{Batch, BatchConfig};
use luna::config::{get_config, base_config, BaseConfig};
use luna::process::Process;
use luna::optimization::{optimize, MainConf};

use model::Model;
use model::network::representation::DefaultRepresentation;

use tasks::{Task, TaskEval};

use evolution::algorithm::nes::NES;

use utils::random;
use utils::logger::init_logger;

use std::env;


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

        let evaluator: MultiEvaluator<T> = Self::evaluator(&conf);
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
