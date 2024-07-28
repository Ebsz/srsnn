use crate::eval::MultiEvaluator;
use crate::process::{Process, MainConf};
use crate::optimization::Optimizer;
use crate::config::{get_config, BaseConfig};
use crate::analysis::run_analysis;

use model::Model;

use evolution::algorithm::Algorithm;
use evolution::algorithm::nes::NES;
use evolution::stats::EvolutionStatistics;

use tasks::{Task, TaskEval};

use ndarray::Array;

use utils::math;
use utils::environment::Environment;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Optimization of config parameters
pub struct HyperOptimization;


impl HyperOptimization {
    //fn algorithm<M: Model, T: Task + TaskEval, A: Algorithm>(
    //    conf: BaseConfig,
    //    main_conf: MainConf<M,A>,
    //    env: Environment,
    //    stop_signal: Arc<AtomicBool>,
    //    params: Vec<(f32, f32)>)
    //    -> Vec<EvolutionStatistics> {

    //

    //    stats
    //}
}

impl Process for HyperOptimization {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let mut main_conf = Self::main_conf::<M, T, NES>();

        Self::log_config(&conf, &main_conf);
        let env = Self::environment::<T>();

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        // Create params
        let alpha = Array::linspace(0.001, 0.01, 2).to_vec();
        let sigma = Array::linspace(0.01, 3.0, 12).to_vec();

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

        report(&mut stats, params.as_slice());

    }
}

fn report(stats: &mut [EvolutionStatistics], param_range: &[(f32, f32)]) {
    log::info!("Optimization report:");


    // Find best eval for each experiment
    let mut best: Vec<f32> = vec![];
    for e in stats {
        let b = math::maxf(&(e.runs.iter().map(|r| math::maxf(&r.best_fitness)).collect()));
        best.push(b);
    }

    //let best: Vec<f32> = stats.iter().map(|s| s.runs math::maxf(&s.runs[0].best_fitness)).collect();

    let z: Vec<(f32, (f32, f32))> = best.iter().zip(param_range).map(|(a,b)| (*a,*b)).collect();

    //log::info!("\nbest | alpha\n{:#?}", z);
    println!("\nbest | (alpha, sigma)");
    for (a,b) in z {
        println!("{:.3} | {:?}", a, b);
    }

    //let r = run_analysis::<T>(&stats.run().best_network[0]);
    //crate::plots::generate_plots(&r);
}
