use crate::eval::{Evaluation, validate_pattern_task};
use crate::process::MainConf;
use crate::analysis::graph::{Graph, GraphAnalysis};

use utils::config::{Configurable, ConfigSection};

use model::Model;
use model::network::representation::DefaultRepresentation;

use tasks::{Task, TaskEval};

use evolution::Evaluate;
use evolution::stats::OptimizationStatistics;
use evolution::algorithm::Algorithm;

use utils::environment::Environment;

use serde::Deserialize;

use ndarray::Array1;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


// Print log every _ generations
const LOG_FREQ: usize = 1;

#[derive(Clone, Debug, Deserialize)]
pub struct OptimizationConfig {
    pub max_generations: usize,
    pub min_gen_before_restart: usize,
}

impl ConfigSection for OptimizationConfig {
    fn name() -> String {
        "optimizer".to_string()
    }
}

pub struct Optimizer;
impl Configurable for Optimizer {
    type Config = OptimizationConfig;
}

impl Optimizer {
    pub fn optimize<M: Model, T: Task + TaskEval, A: Algorithm>(
        mut eval: impl Evaluate<M, DefaultRepresentation>,
        conf: &MainConf<M, A>,
        env: Environment,
        stop_signal: Arc<AtomicBool>)
    -> OptimizationStatistics {
        let mut algo = A::new::<M>(conf.algorithm.clone(), &conf.model, &env);

        let mut stats = OptimizationStatistics::new();

        let mut gen = 0;
        while !stop_signal.load(Ordering::SeqCst)
            && stats.sum_generations() < conf.optimizer.max_generations  {

            let ps = algo.parameter_sets();

            for i in 0..ps.len() {
                assert!(!ps[i].is_nan(), "Error in parameter set: {:#?}", ps[i].set);
            }

            let mut models = vec![];
            for p in ps {
                models.push(M::new(&conf.model, p, &env));
            }

            let e: Vec<(u32, &M)> = models.iter().enumerate()
                .map(|(i, m)| (i as u32, m)).collect();

            let evaluations = eval.eval(&e);
            let fitness: Vec<f32> = evaluations.iter().map(|e| e.1).collect();

            log_generation::<T>(gen, &mut stats, &evaluations);

            algo.step(fitness);

            gen += 1
        }

        log::info!("finished");

        stats
    }
}

fn sorted_fitness(evals: &[Evaluation]) -> Vec<(u32, f32)> {
    let mut sorted_fitness: Vec<(u32, f32)> = evals.iter()
        .map(|g| (g.0, g.1)).collect();

    sorted_fitness.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    assert!(sorted_fitness.windows(2).all(|f| f[0].1 >= f[1].1));

    sorted_fitness
}

fn log_generation<T: Task + TaskEval>(gen: usize, stats: &mut OptimizationStatistics, evals: &[Evaluation]) {
    let sorted = sorted_fitness(evals);

    let scores: Array1<f32> = sorted.iter().map(|(_, f)| *f).collect();

    let fitness_mean: f32 = scores.iter().sum::<f32>() / sorted.len() as f32;
    let fitness_std: f32 = scores.std(0.0);
    let best_fitness: f32 = sorted[0].1;

    let best: &DefaultRepresentation = evals.iter()
        .filter_map(|(i,_,r)| if *i == sorted[0].0 {Some(r)} else {None})
        .collect::<Vec<&DefaultRepresentation>>()[0];

    analyze_model::<T>(best);

    if gen % LOG_FREQ == 0 {
        log::info!("Generation {} - best fit: {:.3}, mean: {:.3}, std: {:.3} - ({}, {})",
            gen, best_fitness, fitness_mean, fitness_std, stats.runs.len(), stats.sum_generations());
    }

    //let val_fitness = validate_pattern_task(best);
    //log::info!("val eval: {val_fitness}");

    stats.log_generation(best_fitness, fitness_mean, best.clone());
}

fn analyze_model<T: Task + TaskEval>(r: &DefaultRepresentation) {
    if log::log_enabled!(log::Level::Trace) {
        log::trace!("Analyzing best network..");

        let g: Graph = r.into();
        let ga = GraphAnalysis::analyze(&g);

        log::trace!("Graph: {}\n{}", g, ga);
    }
}
