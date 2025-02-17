//! Default process.
//!
//! Performs a single optimization run.

use crate::config::BaseConfig;
use crate::analysis::{analyze_network, run_analysis};
use crate::process::{Process, MainConf};
use crate::eval::MultiEvaluator;
use crate::optimization::Optimizer;
use crate::plots;
use crate::plots::plt;
use model::Model;

use tasks::{Task, TaskEval};

use evolution::algorithm::snes::SeparableNES;
use evolution::stats::OptimizationStatistics;

use utils::random;
use utils::environment::Environment;
use model::record::RecordType;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use ndarray::Array2;


pub struct DefaultProcess;
impl Process for DefaultProcess {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        let main_conf = Self::main_conf::<M, T, SeparableNES>();
        let env = Self::environment::<T>();

        Self::log_config(&conf, &main_conf, &env);

        Self::single_run::<M, T>(conf, main_conf, env);
    }
}

impl DefaultProcess {
    fn single_run<M: Model, T: Task + TaskEval>(conf: BaseConfig, main_conf: MainConf<M, SeparableNES>, env: Environment) {
        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        let setups = T::eval_setups();

        let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval, setups.clone());

        let mut stats = Optimizer::optimize::<M, T, SeparableNES>(evaluator,
            &main_conf, env.clone(), stop_signal.clone());

        Self::report::<T>(&mut stats, &conf);
    }

    fn report<T: Task + TaskEval>(stats: &mut OptimizationStatistics, base_config: &BaseConfig) {
        plots::plot_stats(stats, "run");

        let (f, repr, _) = stats.best();
        log::info!("Best fitness: {f}");
        analyze_network(repr);


        let setup = T::eval_setups()[0].clone();
        let record = run_analysis::<T>(repr, &[setup])[0].clone();

        plots::generate_plots(&record);
        plots::plot_run_spikes(&record, None);
        plots::single_neuron_dynamics(&record);

        let s: Array2<u32> = record.as_array(RecordType::Spikes).map(|x| *x as u32);
        let fr = utils::analysis::firing_rate(s, 10);
        let p_series = record.as_array(RecordType::Potentials);

        let _ = plt::plot_matrix(&fr, "firing_rate.png");
        let _ = plt::plot_matrix(&p_series, "potentials.png");

        Self::save(repr.clone(),
            format!("network_{}_{}_{}",
            base_config.model, base_config.task,
            random::random_range((0,1000000)).to_string()));
    }
}
