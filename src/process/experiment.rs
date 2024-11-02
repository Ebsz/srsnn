//! Runs a config a set number of times

use crate::config::{get_config, BaseConfig};
use crate::analysis;
use crate::process::{Process, MainConf};
use crate::eval::MultiEvaluator;
use crate::optimization::Optimizer;
use crate::plots;
use model::Model;

use tasks::{Task, TaskEval};

use evolution::algorithm::snes::SeparableNES;
use evolution::stats::OptimizationStatistics;

use utils::config::{Configurable, ConfigSection};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;


pub struct Experiment;
impl Process for Experiment {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let main_conf = Self::main_conf::<M, T, SeparableNES>();

        let experiment_conf = get_config::<Self>();

        log::info!("Starting experiment with {} runs", experiment_conf.n_runs);
        let experiment_stats = Self::multiple_runs::<M, T>(&conf, main_conf, experiment_conf);

        Self::experiment_report(experiment_stats);
    }
}

impl Experiment {
    fn multiple_runs<M: Model, T: Task + TaskEval>(
        conf: &BaseConfig,
        main_conf: MainConf<M, SeparableNES>,
        experiment_conf: ExperimentConfig)
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

            let stats = Optimizer::optimize::<M, T, SeparableNES>(evaluator,
                &main_conf, env.clone(), stop_signal.clone());

            if experiment_conf.plot_individual_runs {
                Self::run_report::<T>(&stats, n);
            }

            run_stats.push(stats);

            if stop_signal.load(Ordering::SeqCst) {
                break;
            }
        }

        run_stats
    }

    fn run_report<T: Task + TaskEval>(stats: &OptimizationStatistics, n: usize) {
        plots::plot_stats(stats, format!("run_{n}").as_str());

        Self::save::<OptimizationStatistics>(stats.clone(), format!("run_stats_{n}"));
        let (f, repr, _) = stats.best();

        let setup = T::eval_setups()[0].clone();
        let record = analysis::run_analysis::<T>(repr, &[setup])[0].clone();

        plots::plot_run_spikes(&record, Some(format!("spikeplot_{n}").as_str()));
    }

    fn experiment_report(stats: Vec<OptimizationStatistics>) {
        // Merge the stats
        let mut experiment_report = OptimizationStatistics::empty();

        for s in stats {
            println!("n runs: {}", s.runs.len());
            experiment_report.push_run(s.run().clone());
        }

        plots::plot_stats(&experiment_report, format!("experiment").as_str());

        utils::data::save(experiment_report, "experiment_report.json");
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExperimentConfig {
    pub n_runs: usize,
    pub plot_individual_runs: bool,
}

impl Configurable for Experiment {
    type Config = ExperimentConfig;
}

impl ConfigSection for ExperimentConfig {
    fn name() -> String {
        "experiment".to_string()
    }
}
