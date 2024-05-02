//! Contains functionality for performing evals

use crate::phenotype::Phenotype;

use tasks::catching_task::{CatchingTask, CatchingTaskSetup};
use tasks::movement_task::{MovementTask, MovementTaskSetup};
use tasks::survival_task::{SurvivalTask, SurvivalTaskSetup};

use tasks::{Task, TaskName, TaskRenderer};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::{EvolutionEnvironment, Fitness};
use evolution::genome::Genome;


pub fn get_fitness_function(name: TaskName) -> Fitness {
    match name {
        TaskName::CatchingTask => catching_evaluate,
        TaskName::MovementTask => movement_evaluate,
        TaskName::SurvivalTask => survival_evaluate,
    }
}

pub fn survival_evaluate(g: &Genome, env: &EvolutionEnvironment) -> f32 {
    let mut phenotype = Phenotype::from_genome(g, env);

    let task = SurvivalTask::new(SurvivalTaskSetup {});

    let mut runner = TaskRunner::new(task, &mut phenotype);
    let result = runner.run();

    let eval = (result.time as f32 / SurvivalTask::MAX_T as f32) * 100.0;
    log::trace!("eval: {:?}", eval);

    eval
}

pub fn movement_evaluate(g: &Genome, env: &EvolutionEnvironment) -> f32 {
    let mut phenotype = Phenotype::from_genome(g, env);

    let task = MovementTask::new(MovementTaskSetup {});

    let mut runner = TaskRunner::new(task, &mut phenotype);
    let result = runner.run();

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

        let task_conf = CatchingTaskSetup {
            target_pos: trial_positions[i]
        };

        let task = CatchingTask::new(task_conf);

        let mut runner = TaskRunner::new(task, &mut phenotype);
        let result = runner.run();

        total_fitness += (1.0 - result.distance/max_distance) * 100.0 - (if result.success {0.0} else {30.0});
        if result.success {
            correct += 1;
        }
    }

    let fitness = total_fitness / trial_positions.len() as f32;

    log::trace!("eval: {:?} correct: {}/11", fitness, correct);

    fitness
}
