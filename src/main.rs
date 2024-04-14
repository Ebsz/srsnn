//! luna/src/main.rs

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


static DEFAULT_CONF: RunConfig = RunConfig {
    fitness_fn: movement_evaluate,
    taskname: TaskName::MovementTask,
};


#[allow(dead_code)]
fn movement_evaluate(g: &Genome, env: &EvolutionEnvironment) -> f32 {
    let mut phenotype = Phenotype::from_genome(g, env);

    let task = MovementTask::new(MovementTaskConfig {});

    let mut executor = TaskExecutor::new(task, &mut phenotype);
    let result = executor.execute(false);

    let eval = (result.distance as f32 / 600.0) * 100.0;
    log::trace!("eval: {:?}", eval);

    eval
}

#[allow(dead_code)]
fn catching_evaluate(g: &Genome, env: &EvolutionEnvironment) -> f32 {
    let trial_positions: [i32; 11] = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];

    let mut phenotype = Phenotype::from_genome(g, env);

    let max_distance = tasks::catching_task::CatchingTask::render_size().0 as f32;

    let mut total_fitness = 0.0;
    let mut correct: u32 = 0;

    for i in 0..trial_positions.len() {
        phenotype.reset();

        let task_conf = CatchingTaskConfig {
            target_pos: trial_positions[i]
        };

        let task = CatchingTask::new(task_conf);

        let mut executor = TaskExecutor::new(task, &mut phenotype);
        let result = executor.execute(false);

        total_fitness += (1.0 - result.distance/max_distance) * 100.0 - (if result.success {0.0} else {30.0});
        if result.success {
            correct += 1;
        }
    }

    let fitness = total_fitness / trial_positions.len() as f32;

    log::trace!("eval: {:?} correct: {}/11", fitness, correct);

    fitness
}

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
    log::info!("Parsing config");
    log::info!("Task: {:?}", conf.taskname);

    let task_environment = tasks::get_environment(conf.taskname);

    let env = EvolutionEnvironment {
        inputs: task_environment.agent_inputs,
        outputs: task_environment.agent_outputs
    };

    let mut population = Population::new(env.clone(), conf.fitness_fn);

    let evolved_genome: Genome = population.evolve();

    //let task = CatchingTask::new( CatchingTaskConfig {
    //    target_pos: 450
    //});
    //
    let task = MovementTask::new(MovementTaskConfig {});
    visualize_genome_on_task(task, &evolved_genome, &env);
}

fn main() {
    init_logger();
    log::info!("seed is {}", SEED);

    run(&DEFAULT_CONF);
}
