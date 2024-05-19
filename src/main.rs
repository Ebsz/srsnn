//! luna/src/main.rs

use luna::eval::TaskEvaluator;
use luna::phenotype::EvolvableGenome;

use luna::visual::plots::{plot_evolution_stats};
use luna::config::{get_config, genome_config, MainConfig};

use luna::models::stochastic::random_model::RandomGenome;
use luna::models::stochastic::base_model::BaseStochasticGenome;

use evolution::EvolutionEnvironment;
use evolution::population::Population;

use evolution::genome::matrix_genome::MatrixGenome;

use tasks::{Task, TaskEval};
use tasks::catching_task::CatchingTask;
use tasks::movement_task::MovementTask;
use tasks::survival_task::SurvivalTask;
use tasks::energy_task::EnergyTask;
use tasks::xor_task::XORTask;

use utils::logger::init_logger;
use utils::random::SEED;

use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


trait Process {
    type Config;

    fn run(conf: &Self::Config);
}

struct EvolutionProcess;

impl EvolutionProcess {
    fn resolve(config: &MainConfig) {
        match config.genome.as_str() {
            "matrix" => { Self::resolve_t::<MatrixGenome>(config); },
            "random" => { Self::resolve_t::<RandomGenome>(config); },
            "base_stochastic" => { Self::resolve_t::<BaseStochasticGenome>(config); },
            _ => {panic!("Unknown genome: {}", config.genome);}
        }
    }

    fn resolve_t<G: EvolvableGenome>(config: &MainConfig) {
        match config.task.as_str() {
            "catching" => { Self::evolve::<G, CatchingTask>(config); },
            "movement" => { Self::evolve::<G, MovementTask>(config); },
            "survival" => { Self::evolve::<G, SurvivalTask>(config); },
            "energy"   => { Self::evolve::<G, EnergyTask>(config); },
            "xor"      => { Self::evolve::<G, XORTask>(config); },
            _ => { panic!("Unknown task: {}", config.task); }
        }
    }

    fn evolve<G: EvolvableGenome, T: Task + TaskEval>(config: &MainConfig) {
        log::info!("task: {}", config.task);
        log::info!("genome: {}", config.genome);

        let env = Self::environment::<T>();

        let evaluator = TaskEvaluator::<T, G>::new(env.clone());

        let genome_config = genome_config::<G>();

        let mut population = Population::new(env.clone(), config.evolution, genome_config, evaluator);

        init_ctrl_c_handler(population.stop_signal.clone());

        let _evolved_genome = population.evolve();

        plot_evolution_stats(&population.stats);

        //let task = T::new(&T::eval_setups()[0]);
        //visualize_genome_on_task(task, evolved_genome, &env);
    }

    fn environment<T: Task>() -> EvolutionEnvironment {
        let e = T::environment();

        EvolutionEnvironment {
            inputs: e.agent_inputs,
            outputs: e.agent_outputs,
        }
    }
}



///// Analyzes a genome resulting from an evolutionary process
//#[allow(dead_code)]
//fn analyze_genome(g: &Genome, env: &EvolutionEnvironment) {
//    log::info!("Analyzing genome");
//    let mut phenotype = Phenotype::from_genome(g, env);
//    phenotype.enable_recording();
//
//    let task = CatchingTask::new( CatchingTaskSetup {
//        target_pos: 450
//    });
//
//    let mut runner = TaskRunner::new(task, &mut phenotype);
//
//    runner.run();
//
//    generate_plots(&phenotype.record);
//}


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

impl Process for EvolutionProcess {
    type Config = MainConfig;

    fn run(config: &MainConfig) {
        Self::resolve(config);
    }
}


struct TestProcess;

impl Process for TestProcess {
    type Config = bool;

    fn run(_config: &Self::Config) {
        log::info!("Running test process");
    }
}

fn main() {
    let config_name = parse_config_name_from_args();
    let config = get_config(config_name.clone());

    init_logger(config.log_level.clone());

    log::info!("Using config: {}", config_name.unwrap_or("default".to_string()));
    log::info!("seed is {}", SEED);

    EvolutionProcess::run(&config);
    //TestProcess::run(&true);
}
