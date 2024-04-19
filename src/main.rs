//! luna/src/main.rs

use luna::evaluate;
use luna::phenotype::Phenotype;
use luna::task_executor::TaskExecutor;
use luna::visual::plots::generate_plots;
use luna::visual::visualize_genome_on_task;
use luna::config::RunConfig;

use evolution::genome::Genome;
use evolution::{Population, EvolutionEnvironment};

use tasks::{Task, TaskName, TaskRenderer};
use tasks::catching_task::{CatchingTask, CatchingTaskConfig};
use tasks::movement_task::{MovementTask, MovementTaskConfig};

use utils::logger::init_logger;
use utils::random::SEED;


/// Analyzes a genome resulting from an evolutionary process
#[allow(dead_code)]
fn analyze_genome(g: &Genome, env: &EvolutionEnvironment) {
    log::info!("Analyzing genome");
    let mut phenotype = Phenotype::from_genome(g, env);

    let task = CatchingTask::new( CatchingTaskConfig {
        target_pos: 450
    });

    let mut executor = TaskExecutor::new(task, &mut phenotype);

    executor.execute(true);
    generate_plots(&executor.record);
}

fn run(conf: &RunConfig) {
    log::info!("Task: {:?}", conf.task);

    let task_environment = tasks::get_environment(conf.task);

    let env = EvolutionEnvironment {
        inputs: task_environment.agent_inputs,
        outputs: task_environment.agent_outputs
    };

    let fitness_function = evaluate::get_fitness_function(conf.task);

    let mut population = Population::new(env.clone(), fitness_function, conf.evolution_config);

    let evolved_genome: Genome = population.evolve();

    let task = CatchingTask::new(CatchingTaskConfig {
        target_pos: 450
    });

    visualize_genome_on_task(task, &evolved_genome, &env);
}

fn main() {
    init_logger();
    log::info!("seed is {}", SEED);

    run(&RunConfig::default());
}
