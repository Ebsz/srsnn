//! luna/src/main.rs

use luna::eval::TaskEvaluator;
use luna::phenotype::EvolvableGenome;

use luna::visual::plots::{plot_evolution_stats};
use luna::config::{get_config, genome_config, MainConfig};

use luna::models::stochastic::random_model::RandomGenome;
use luna::models::stochastic::base_model::BaseStochasticGenome;
use luna::models::matrix::MatrixGenome;

use evolution::EvolutionEnvironment;
use evolution::population::Population;

use tasks::{Task, TaskEval};
use tasks::task_runner::TaskRunner;
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
    fn run<G: EvolvableGenome, T: Task + TaskEval>(conf: MainConfig);

    fn init(config: MainConfig) {
        match config.genome.as_str() {
            "matrix" => { Self::resolve_t::<MatrixGenome>(config); },
            "random" => { Self::resolve_t::<RandomGenome>(config); },
            "base_stochastic" => { Self::resolve_t::<BaseStochasticGenome>(config); },
            _ => {panic!("Unknown genome: {}", config.genome);}
        }
    }

    fn resolve_t<G: EvolvableGenome>(config: MainConfig) {
        match config.task.as_str() {
            "catching" => { Self::run::<G, CatchingTask>(config); },
            "movement" => { Self::run::<G, MovementTask>(config); },
            "survival" => { Self::run::<G, SurvivalTask>(config); },
            "energy"   => { Self::run::<G, EnergyTask>(config); },
            "xor"      => { Self::run::<G, XORTask>(config); },
            _ => { panic!("Unknown task: {}", config.task); }
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
    fn run<G: EvolvableGenome, T: Task + TaskEval>(config: MainConfig) {
        log::info!("task: {}", config.task);
        log::info!("genome: {}", config.genome);

        let env = Self::environment::<T>();

        let evaluator = TaskEvaluator::<T, G>::new(env.clone());

        let genome_config = genome_config::<G>();

        let mut population = Population::new(env.clone(), config.evolution, genome_config, evaluator);

        init_ctrl_c_handler(population.stop_signal.clone());

        let evolved_genome = population.evolve();

        plot_evolution_stats(&population.stats);

        //let task = T::new(&T::eval_setups()[0]);
        //visualize_genome_on_task(task, evolved_genome, &env);
    }
}

struct AnalysisProcess;

impl Process for AnalysisProcess {
    fn run<G: EvolvableGenome, T: Task + TaskEval>(config: MainConfig) {
        log::info!("Running analysis");

        let genome_config = genome_config::<G>();
        let env = Self::environment::<T>();

        let genome = G::new(&env, &genome_config);
        let setups = T::eval_setups();

        let task = T::new(&setups[0]);

        let mut p = genome.to_phenotype(&env);
        let mut runner = TaskRunner::new(task, &mut p);
        runner.run();
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


fn main() {
    let config_name = parse_config_name_from_args();
    let config = get_config(config_name.clone());

    init_logger(config.log_level.clone());

    log::info!("Using config: {}", config_name.unwrap_or("default".to_string()));
    log::info!("seed is {}", SEED);

    EvolutionProcess::init(config);
    //AnalysisProcess::init(config);
}
