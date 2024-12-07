//! Hyperparameter optimization using SNES.
//!
//! Fitness function is given by f: setup -> R := setup-best-fitness / gen

use crate::plots;
use crate::eval::MultiEvaluator;
use crate::process::Process;
use crate::optimization::Optimizer;
use crate::config::BaseConfig;
use crate::analysis;

use model::Model;
use model::network::representation::DefaultRepresentation;

use evolution::algorithm::snes::SeparableNES;
use evolution::stats::OptimizationStatistics;

use tasks::{Task, TaskEval};

use ndarray::array;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


pub struct HyperOptimization;

impl Process for HyperOptimization {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let mut main_conf = Self::main_conf::<M, T, SeparableNES>();
        let env = Self::environment::<T>();

        Self::log_config(&conf, &main_conf, &env);

        // Create params
        let lr_mu = array![0.1, 0.2, 0.3, 0.4, 0.5]; //Array::linspace(0.1, 1.5, 10).to_vec();
        let lr_sigma = array![0.01, 0.05, 0.1];  //Array::linspace(0.001, 0.01, 2).to_vec();

        // Cartesian of the parameters
        let params: Vec<(f32, f32)> = lr_mu.iter()
            .flat_map(|&x| std::iter::repeat(x).zip(lr_sigma.clone())).collect();

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        log::info!("Starting hyperparameter search over {} parameter sets", params.len());

        let setups = T::eval_setups();

        let mut stats = vec![];
        for p in &params {
            main_conf.algorithm.lr_mu = p.0;
            main_conf.algorithm.lr_sigma = p.1;
            //main_conf.eval.trials = p.2;

            log::info!("lr_mu: {}, lr_sigma: {}", main_conf.algorithm.lr_mu, main_conf.algorithm.lr_sigma);

            let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval, setups.clone());
            let s = Optimizer::optimize::<M, T, SeparableNES>(evaluator, &main_conf, env.clone(), stop_signal.clone());
            stats.push(s);

            if stop_signal.load(Ordering::SeqCst) {
                log::info!("Ending");
                break;
            }
        }

        Self::hyper_report::<T>(&mut stats, params.as_slice());
    }
}

impl HyperOptimization {
    fn hyper_report<T: Task + TaskEval>(stats: &mut [OptimizationStatistics], param_range: &[(f32, f32)]) {
        log::info!("Experiment report:");

        // Best eval for each experiment
        let best: Vec<(f32, &DefaultRepresentation, _)> = stats.iter().map(|x| x.best()).collect();

        let z: Vec<(f32, (f32, f32))> = best.iter().map(|x| x.0).zip(param_range)
                                            .map(|(a,b)| (a,*b)).collect();

        println!("\nbest | (alpha, sigma)");
        for (a,b) in z {
            println!("{:.3} | {:?}", a, b);
        }

        let (e, repr, _) = stats.iter().map(
            |x| x.best()).max_by(|a,b| a.0.partial_cmp(&b.0).expect("")).unwrap();

        println!("Best eval: {e}");
        analysis::analyze_network(&repr);

        Self::save(repr.clone(),
            format!("experiment_{}", utils::random::random_range((0,1000000)).to_string()));
    }
}
