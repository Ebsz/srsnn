use crate::config::BaseConfig;
use crate::analysis::run_analysis;
use crate::process::Process;
use crate::eval::MultiEvaluator;
use crate::optimization::Optimizer;
use crate::plots;
use model::Model;

use tasks::{Task, TaskEval};

//use evolution::algorithm::nes::NES;
use evolution::algorithm::snes::SeparableNES;
use evolution::stats::EvolutionStatistics;

use utils::random;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;


pub struct OptimizationProcess;
impl Process for OptimizationProcess {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let main_conf = Self::main_conf::<M, T, SeparableNES>();

        Self::log_config(&conf, &main_conf);

        let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval);
        let env = Self::environment::<T>();

        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        let mut stats = Optimizer::optimize::<M, T, SeparableNES>(evaluator, &main_conf, env, stop_signal);

        Self::report::<T>(&mut stats, &conf);
    }
}

impl OptimizationProcess {
    fn report<T: Task + TaskEval>(stats: &mut EvolutionStatistics, base_config: &BaseConfig) {
        plots::plot_evolution_stats_all(stats);

        let (f, repr) = stats.best();
        log::info!("Best fitness: {f}");
        Self::analyze_network(repr);

        let record = run_analysis::<T>(&repr);
        plots::generate_plots(&record);

        Self::save_network(repr.clone(),
            format!("network_{}_{}_{}",
            base_config.model, base_config.task,
            random::random_range((0,1000000)).to_string()));
    }
}
