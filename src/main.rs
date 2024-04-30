//! luna/src/main.rs

use luna::evaluate;
use luna::phenotype::Phenotype;
use luna::visual::plots::generate_plots;
use luna::visual::visualize_genome_on_task;
use luna::config::{MainConfig, get_config, get_taskname};

use evolution::EvolutionEnvironment;
use evolution::population::Population;
use evolution::genome::Genome;

use tasks::Task;
use tasks::task_runner::{TaskRunner};
use tasks::catching_task::{CatchingTask, CatchingTaskConfig};

use utils::logger::init_logger;
use utils::random::SEED;

use std::sync::atomic::Ordering;
use std::env;


/// Analyzes a genome resulting from an evolutionary process
#[allow(dead_code)]
fn analyze_genome(g: &Genome, env: &EvolutionEnvironment) {
    log::info!("Analyzing genome");
    let mut phenotype = Phenotype::from_genome(g, env);
    phenotype.enable_recording();

    let task = CatchingTask::new( CatchingTaskConfig {
        target_pos: 450
    });

    let mut runner = TaskRunner::new(task, &mut phenotype);

    runner.run();

    generate_plots(&phenotype.record);
}

fn init_ctrl_c_handler(population: &Population) {
    let stop_signal = population.stop_signal.clone();
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

fn run(conf: &MainConfig) {
    let task_name = get_taskname(&conf.task.name);
    log::info!("Task: {:?}", task_name);

    let task_environment = tasks::get_environment(task_name);

    let env = EvolutionEnvironment {
        inputs: task_environment.agent_inputs,
        outputs: task_environment.agent_outputs,
        fitness: evaluate::get_fitness_function(task_name)
    };

    let mut population = Population::new(env.clone(), conf.evolution, conf.genome);

    init_ctrl_c_handler(&population);

    let _evolved_genome: Genome = population.evolve();
}

fn parse_config_path_from_args() -> Option<String> {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        return Some(args[1].clone());
    }

    None
}

fn main() {
    init_logger();

    let config_path = parse_config_path_from_args();
    let config = get_config(config_path);

    log::info!("seed is {}", SEED);
    run(&config);
}
