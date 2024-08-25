//! Default process.
//!
//! Performs a single optimization run.

use crate::config::BaseConfig;
use crate::analysis::{analyze_network, run_analysis};
use crate::process::{Process, MainConf};
use crate::eval::MultiEvaluator;
use crate::optimization::Optimizer;
use crate::plots;
use model::Model;

use tasks::{Task, TaskEval};

use evolution::algorithm::snes::SeparableNES;
use evolution::stats::OptimizationStatistics;

use utils::random;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


pub struct DefaultProcess;
impl Process for DefaultProcess {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let main_conf = Self::main_conf::<M, T, SeparableNES>();

        Self::single_run::<M, T>(conf, main_conf);
    }
}

impl DefaultProcess {
    fn report<T: Task + TaskEval>(stats: &mut OptimizationStatistics, base_config: &BaseConfig) {
        plots::plot_stats(stats, "run");

        let (f, repr, _) = stats.best();
        log::info!("Best fitness: {f}");
        analyze_network(repr);

        let record = run_analysis::<T>(&repr);
        plots::generate_plots(&record);

        Self::save(repr.clone(),
            format!("network_{}_{}_{}",
            base_config.model, base_config.task,
            random::random_range((0,1000000)).to_string()));
    }

    fn single_run<M: Model, T: Task + TaskEval>(conf: BaseConfig, main_conf: MainConf<M, SeparableNES>) {
        let env = Self::environment::<T>();

        Self::log_config(&conf, &main_conf, &env);

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        let setups = T::eval_setups();

        let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval, setups.clone());

        let mut stats = Optimizer::optimize::<M, T, SeparableNES>(evaluator,
            &main_conf, env.clone(), stop_signal.clone());

        Self::report::<T>(&mut stats, &conf);
    }

    fn multiple_runs<M: Model, T: Task + TaskEval>(conf: BaseConfig, main_conf: MainConf<M, SeparableNES>, n_runs: usize) {
        let env = Self::environment::<T>();

        Self::log_config(&conf, &main_conf, &env);

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        let setups = T::eval_setups();

        for n in 0..n_runs {
            log::info!("Run {n}");
            let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval, setups.clone());

            let mut stats = Optimizer::optimize::<M, T, SeparableNES>(evaluator,
                &main_conf, env.clone(), stop_signal.clone());

            Self::report::<T>(&mut stats, &conf);

            if stop_signal.load(Ordering::SeqCst) {
                break;
            }
        }
    }
}
