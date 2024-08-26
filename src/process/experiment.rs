//! Runs a config a set number of times

use crate::config::{get_config, BaseConfig};
use crate::analysis::run_analysis;
use crate::process::{Process, MainConf};
use crate::eval::MultiEvaluator;
use crate::optimization::Optimizer;
use crate::plots;
use model::Model;

use tasks::{Task, TaskEval};

use evolution::algorithm::snes::SeparableNES;
use evolution::stats::OptimizationStatistics;

use utils::random;
use utils::config::{Configurable, ConfigSection};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;


#[derive(Clone, Debug, Deserialize)]
pub struct ExperimentConfig {
    pub n_runs: usize
}

impl Configurable for Experiment {
    type Config = ExperimentConfig;
}

impl ConfigSection for ExperimentConfig {
    fn name() -> String {
        "experiment".to_string()
    }
}

pub struct Experiment;
impl Process for Experiment {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let main_conf = Self::main_conf::<M, T, SeparableNES>();

        let experiment_conf = get_config::<Experiment>();

        log::info!("Starting experiment with {} runs", experiment_conf.n_runs);
        let mut stats = Self::multiple_runs::<M, T>(&conf, main_conf, experiment_conf);
    }
}

impl Experiment {
    fn run_report<T: Task + TaskEval>(stats: &OptimizationStatistics, n: usize) {
        plots::plot_stats(stats, format!("run_{n}").as_str());

        Self::save::<OptimizationStatistics>(stats.clone(), format!("run_stats_{n}"));
    }

    // Multiple runs over the same config
    fn multiple_runs<M: Model, T: Task + TaskEval>(conf: &BaseConfig, main_conf: MainConf<M, SeparableNES>, experiment_conf: ExperimentConfig)
    -> Vec<OptimizationStatistics> {
        let env = Self::environment::<T>();

        Self::log_config(&conf, &main_conf, &env);

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        let setups = T::eval_setups();

        let mut run_stats: Vec<OptimizationStatistics> = Vec::new();

        for n in 0..experiment_conf.n_runs {
            log::info!("Run {n}");
            let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval, setups.clone());

            let mut stats = Optimizer::optimize::<M, T, SeparableNES>(evaluator,
                &main_conf, env.clone(), stop_signal.clone());

            Self::run_report::<T>(&stats, n);

            run_stats.push(stats);

            if stop_signal.load(Ordering::SeqCst) {
                break;
            }
        }

        run_stats
    }
}
