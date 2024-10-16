use crate::eval::{run_network_on_task, Evaluation, MultiEvaluator};
use crate::process::MainConf;
use crate::analysis::graph::{Graph, GraphAnalysis};

use utils::config::{Configurable, ConfigSection};
use utils::parameters::ParameterSet;

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

const VALIDATION_FREQ: usize = 1;

pub struct Optimizer;
impl Configurable for Optimizer {
    type Config = OptimizationConfig;
}

impl Optimizer {
    pub fn optimize<M: Model, T: Task + TaskEval, A: Algorithm>(
        mut eval: MultiEvaluator<T>,
        conf: &MainConf<M, A>,
        env: Environment,
        stop_signal: Arc<AtomicBool>)
    -> OptimizationStatistics {

        let mut algo = A::new(conf.algorithm.clone(), M::params(&conf.model, &env));

        let mut stats = OptimizationStatistics::new();

        let mut gen = 0;
        while !stop_signal.load(Ordering::SeqCst)
            && stats.sum_generations() < conf.optimizer.max_generations  {

            let ps = algo.parameter_sets();

            for i in 0..ps.len() {
                assert!(!ps[i].is_nan(), "Error in parameter set: {:#?}", ps[i].set);
            }

            // Create models
            let mut models = vec![];
            for p in ps {
                models.push(M::new(&conf.model, p, &env));
            }

            let e: Vec<(u32, &M)> = models.iter().enumerate()
                .map(|(i, m)| (i as u32, m)).collect();

            let evaluations = eval.eval(&e);
            let fitness: Vec<f32> = evaluations.iter().map(|e| e.1).collect();

            log_generation::<T>(gen, &mut stats, &evaluations, &ps, &eval);

            if Array1::<f32>::from_vec(fitness.clone()).std(0.0) == 0.0 {
               log::warn!("cannot optimize because eval stddev was 0.0, trying again");

               //algo = A::new(conf.algorithm.clone(), M::params(&conf.model, &env));
               //stats.new_run();

               continue;
            }

            algo.step(fitness);

            gen += 1
        }

        stats
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct OptimizationConfig {
    pub max_generations: usize,
}

impl ConfigSection for OptimizationConfig {
    fn name() -> String {
        "optimizer".to_string()
    }
}

fn sorted_fitness(evals: &[Evaluation]) -> Vec<(u32, f32)> {
    let mut sorted_fitness: Vec<(u32, f32)> = evals.iter()
        .map(|g| (g.0, g.1)).collect();

    sorted_fitness.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    assert!(sorted_fitness.windows(2).all(|f| f[0].1 >= f[1].1));

    sorted_fitness
}

fn log_generation<T: Task + TaskEval>(
    gen: usize,
    stats: &mut OptimizationStatistics,
    evals: &[Evaluation],
    ps: &[ParameterSet],
    eval: &MultiEvaluator<T>)
{
    let sorted = sorted_fitness(evals);

    let scores: Array1<f32> = sorted.iter().map(|(_, f)| *f).collect();

    let fitness_mean: f32 = scores.iter().sum::<f32>() / sorted.len() as f32;
    let fitness_std: f32 = scores.std(0.0);
    let best_fitness: f32 = sorted[0].1;

    let best_repr: &DefaultRepresentation = evals.iter()
        .filter_map(|(i,_,r)| if *i == sorted[0].0 {Some(r)} else {None})
        .collect::<Vec<&DefaultRepresentation>>()[0];

    let best_ps: &ParameterSet = ps.iter().enumerate()
        .filter_map(|(i, p)| if i as u32 == sorted[0].0 { Some(p) } else { None } )
        .collect::<Vec<&ParameterSet>>()[0];

    analyze_model::<T>(best_repr);

    if gen % LOG_FREQ == 0 {
        log::info!("Gen. {} - best fit: {:.3}, mean: {:.3}, std: {:.3}",
            gen, best_fitness, fitness_mean, fitness_std);
    }

    // Perform validation on the best network
    if gen % VALIDATION_FREQ == 0 {
        validation(best_repr, eval, stats);
    }

    stats.log_generation(best_fitness, fitness_mean, fitness_std, (best_repr.clone(), best_ps.clone()));
}

fn analyze_model<T: Task + TaskEval>(r: &DefaultRepresentation) {
    if log::log_enabled!(log::Level::Trace) {
        log::trace!("Analyzing best network..");

        let g: Graph = r.into();
        let ga = GraphAnalysis::analyze(&g);

        log::trace!("Graph: {}\n{}", g, ga);
    }
}

fn validation<T: Task + TaskEval>(
    r: &DefaultRepresentation,
    eval: &MultiEvaluator<T>,
    stats: &mut OptimizationStatistics
    ) {
    let validation_setups = eval.validation_setups();

    if validation_setups.len() != 0 {
        let results = run_network_on_task::<T>(r, validation_setups);

        let accuracy = T::accuracy(&results);
        let val = T::fitness(results);

        match accuracy {
            Some(acc) =>  {
                log::debug!("validation: {:.3}, accuracy: {:.3}", val, acc);
                stats.log_accuracy(acc);
            },
            None => { log::debug!("validation: {:.3}", val); }
        }

        stats.log_validation(val);
    }
}
