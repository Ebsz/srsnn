//! luna/src/main.rs

use luna::logger::init_logger;

use luna::evolution::genome::Genome;
use luna::evolution::phenotype::Phenotype;
use luna::evolution::{Population, EvolutionEnvironment};

use luna::task_executor::TaskExecutor;
use luna::visual::window::TaskWindow;
use luna::plots::generate_plots;
use luna::utils::SEED;

use tasks::cognitive_task::CognitiveTask;
use tasks::catching_task::{CatchingTask, CatchingTaskConfig};


fn evaluate(g: &Genome, env: &EvolutionEnvironment) -> f32 {
    let trial_positions: [i32; 11] = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];

    let mut phenotype = Phenotype::from_genome(g, env);

    let max_distance = tasks::catching_task::ARENA_SIZE.0 as f32;

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

#[allow(dead_code)]
fn visualize_genome_on_task(g: &Genome, env: &EvolutionEnvironment) {
    log::info!("Visualizing genome behavior on task");

    let task = CatchingTask::new( CatchingTaskConfig {
        target_pos: 450
    });

    let mut phenotype = Phenotype::from_genome(g, env);
    let executor = TaskExecutor::new(task, &mut phenotype);

    let mut window = TaskWindow::new(executor);
    window.run();
}

fn evolve() {
    let task_environment = CatchingTask::environment();

    let env = EvolutionEnvironment {
        inputs: task_environment.agent_inputs,
        outputs: task_environment.agent_outputs
    };

    let mut population = Population::new(env.clone(), evaluate);

    log::info!("Evolving..");
    let evolved_genome: Genome = population.evolve();

    visualize_genome_on_task(&evolved_genome, &env);
    //analyze_genome(&evolved_genome, &env);
}

fn main() {
    init_logger();
    log::info!("seed is {}", SEED);

    evolve();
}
