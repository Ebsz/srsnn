//! luna/src/main.rs

use luna::evaluate;
use luna::phenotype::Phenotype;
use luna::visual::plots::generate_plots;
use luna::config::RunConfig;

use evolution::EvolutionEnvironment;
use evolution::population::Population;
use evolution::genome::Genome;

use tasks::Task;
use tasks::task_runner::{TaskRunner};
use tasks::catching_task::{CatchingTask, CatchingTaskConfig};

use utils::logger::init_logger;
use utils::random::SEED;


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

fn run(conf: &RunConfig) {
    log::info!("Task: {:?}", conf.task);

    let task_environment = tasks::get_environment(conf.task);

    let env = EvolutionEnvironment {
        inputs: task_environment.agent_inputs,
        outputs: task_environment.agent_outputs,
        fitness: evaluate::get_fitness_function(conf.task)
    };

    let mut population = Population::new(env.clone(), conf.evolution_config);

    let _evolved_genome: Genome = population.evolve();
}

fn main() {
    init_logger();
    log::info!("seed is {}", SEED);

    run(&RunConfig::default());
}
