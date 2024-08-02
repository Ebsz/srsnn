//! Hyperparameter optimization

use crate::plots;
use crate::eval::MultiEvaluator;
use crate::process::Process;
use crate::optimization::Optimizer;
use crate::config::BaseConfig;
use crate::analysis::run_analysis;

use model::Model;
use model::network::representation::DefaultRepresentation;

use evolution::algorithm::nes::NES;
use evolution::stats::EvolutionStatistics;

use tasks::{Task, TaskEval};

use ndarray::Array;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


/// Optimization of config parameters
pub struct HyperOptimization;

impl Process for HyperOptimization {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let mut main_conf = Self::main_conf::<M, T, NES>();

        let env = Self::environment::<T>();
        Self::log_config(&conf, &main_conf, &env);

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        // Create params
        let alpha = Array::linspace(0.001, 0.01, 2).to_vec();
        let sigma = Array::linspace(0.1, 1.5, 10).to_vec();

        // Cartesian of the parameters
        let params: Vec<(f32, f32)> = alpha.iter()
            .flat_map(|&x| std::iter::repeat(x).zip(sigma.clone())).collect();

        log::info!("Starting hyperparameter search over {} parameter sets", params.len());
        let mut stats = vec![];
        for p in &params {
            main_conf.algorithm.alpha = p.0;
            main_conf.algorithm.sigma = p.1;

            log::info!("alpha: {}, sigma: {}", main_conf.algorithm.alpha, main_conf.algorithm.sigma);

            let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval);
            let s = Optimizer::optimize::<M, T, NES>(evaluator, &main_conf, env.clone(), stop_signal.clone());
            stats.push(s);

            if stop_signal.load(Ordering::SeqCst) {
                log::info!("Ending");
                break;
            }
        }

        Self::experiment_report::<T>(&mut stats, params.as_slice());
    }
}

impl HyperOptimization {
    fn experiment_report<T: Task + TaskEval>(stats: &mut [EvolutionStatistics], param_range: &[(f32, f32)]) {
        log::info!("Experiment report:");

        // Best eval for each experiment
        let mut best: Vec<(f32, &DefaultRepresentation)> = stats.iter().map(|x| x.best()).collect();

        let z: Vec<(f32, (f32, f32))> = best.iter().map(|x| x.0).zip(param_range)
                                            .map(|(a,b)| (a,*b)).collect();

        println!("\nbest | (alpha, sigma)");
        for (a,b) in z {
            println!("{:.3} | {:?}", a, b);
        }

        let (e, repr) = stats.iter().map(
            |x| x.best()).max_by(|a,b| a.0.partial_cmp(&b.0).expect("")).unwrap();

        println!("Best eval: {e}");
        Self::analyze_network(&repr);

        let record = run_analysis::<T>(&repr);
        plots::generate_plots(&record);


        Self::save_network(repr.clone(),
            format!("experiment_{}", utils::random::random_range((0,1000000)).to_string()));
    }

}
