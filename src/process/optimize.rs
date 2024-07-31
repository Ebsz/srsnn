use crate::config::BaseConfig;
use crate::process::Process;
use crate::eval::MultiEvaluator;
use crate::optimization::Optimizer;
use crate::plots;
use model::Model;

use tasks::{Task, TaskEval};

use evolution::algorithm::nes::NES;
use evolution::stats::EvolutionStatistics;

use utils::random;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;


pub struct OptimizationProcess;
impl Process for OptimizationProcess {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let main_conf = Self::main_conf::<M, T, NES>();

        Self::log_config(&conf, &main_conf);

        let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval);
        let env = Self::environment::<T>();

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        let mut stats = Optimizer::optimize::<M, T, NES>(evaluator, &main_conf, env, stop_signal);

        Self::report(&mut stats, &conf);
    }
}

impl OptimizationProcess {
    fn report(stats: &mut EvolutionStatistics, base_config: &BaseConfig) {
        plots::plot_evolution_stats_all(stats);

        let (_, repr) = stats.best();
        Self::analyze_network(repr);

        Self::save_network(repr.clone(),
            format!("network_{}_{}_{}.json",
            base_config.model, base_config.task,
            random::random_range((0,1000000)).to_string()));
    }
}
