/// Runs the configuration over a number of different runs, saving the best run

use crate::process::Process;
use crate::config::BaseConfig;
use crate::eval::MultiEvaluator;

use crate::optimization::Optimizer;

use evolution::stats::OptimizationStatistics;
use evolution::algorithm::snes::SeparableNES;

use utils::random;
use utils::math;

use tasks::{Task, TaskEval};

use model::Model;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::analysis::run_analysis;


const N_RUNS: usize = 50;

pub struct ExperimentProcess;
impl Process for ExperimentProcess {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        log::info!("Running experiment");


        let mut main_conf = Self::main_conf::<M, T, SeparableNES>();

        let env = Self::environment::<T>();

        Self::log_config(&conf, &main_conf, &env);
        let stop_signal = Arc::new(AtomicBool::new(false));
        Self::init_ctrl_c_handler(stop_signal.clone());

        let mut run_stats = vec![];

        for n in 0..N_RUNS {
            let evaluator: MultiEvaluator<T> = Self::evaluator(&conf, &main_conf.eval);
            let mut stats = Optimizer::optimize::<M, T, SeparableNES>(
                evaluator, &main_conf, env.clone(), stop_signal.clone());

            let f = Self::report::<T>(&mut stats, &conf, n);

            run_stats.push((f, stats));

            if stop_signal.load(Ordering::SeqCst) {
                break;
            }
        }

        let run_fitness: Vec<f32> = run_stats.iter().map(|(f, s)| *f).collect();

        let best_run = &run_stats[math::max_index(run_fitness)];

        Self::save(best_run, format!("best_stats"));
    }
}

impl ExperimentProcess {
    fn report<T: Task + TaskEval>(stats: &mut OptimizationStatistics, base_config: &BaseConfig, n: usize) -> f32 {
        log::info!("Run {n} finished");


        let (f, repr) = stats.best();
        log::info!("Best fitness: {f}");
        Self::analyze_network(repr);

        //let record = run_analysis::<T>(&repr);
        //plots::generate_plots(&record);


        f
    }
}
