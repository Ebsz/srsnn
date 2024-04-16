//! Contains functionality for performing evals

use crate::phenotype::Phenotype;
use crate::task_executor::TaskExecutor;

use tasks::catching_task::{CatchingTask, CatchingTaskConfig};
use tasks::movement_task::{MovementTask, MovementTaskConfig};

use tasks::{Task, TaskRenderer};

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;


pub fn movement_evaluate(g: &Genome, env: &EvolutionEnvironment) -> f32 {
    let mut phenotype = Phenotype::from_genome(g, env);

    let task = MovementTask::new(MovementTaskConfig {});

    let mut executor = TaskExecutor::new(task, &mut phenotype);
    let result = executor.execute(false);

    let eval = (result.distance as f32 / 600.0) * 100.0;
    log::trace!("eval: {:?}", eval);

    eval
}

pub fn catching_evaluate(g: &Genome, env: &EvolutionEnvironment) -> f32 {
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
