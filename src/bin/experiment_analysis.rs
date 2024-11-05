/// Analysis and plotting of experiment reports.

use srsnn::plots;
use srsnn::config;
use srsnn::analysis;
use srsnn::eval;
use srsnn::models::srsnn::base_model::BaseModel;
use srsnn::models::rsnn::RSNNModel;

use srsnn::process::experiment::report::ExperimentReport;

use model::network::representation::DefaultRepresentation;
use evolution::stats::OptimizationStatistics;

use tasks::{Task, TaskEval};
use tasks::pattern::PatternTask;
use tasks::multipattern::MultiPatternTask;
use tasks::pattern_similarity::PatternSimilarityTask;

use utils::data;
use utils::math;
use utils::logger;
use utils::parameters::ParameterSet;

use ndarray::Array2;

use std::env;


fn plot_individual_runs(stats: &OptimizationStatistics) {
    for (i, r) in stats.runs.iter().enumerate() {
        let mut s = OptimizationStatistics::empty();
        s.push_run(r.clone());

        //plots::plot_stats(&s, format!("run_{}", i).as_str());
        plots::plot_run(&r, format!("run_{}", i).as_str());
    }
}

fn analyze_run<T: Task + TaskEval> (repr: &DefaultRepresentation) {
    let setup = T::eval_setups()[4].clone();
    let record = analysis::run_analysis::<T>(repr, &[setup])[0].clone();

    plots::plot_run_spikes(&record, None);
    plots::single_neuron_dynamics(&record);
}

fn accuracy<T: Task + TaskEval> (repr: &DefaultRepresentation) {
    let validation_setups = T::eval_setups();

    let results = eval::run_network_on_task::<T>(repr, &validation_setups);

    let accuracy = T::accuracy(&results);
    let val = T::fitness(results);

    match accuracy {
        Some(acc) =>  {
            println!("validation: {:.3}, accuracy: {:.3}", val, acc);
        },
        None => { println!("validation: {:.3}", val); }
    }
}

fn normalize(d: Array2<f32>) -> Array2<f32> {
    d
        .mapv(|x| if x < 0.0001 { 0.0 } else { x })
        .mapv(|x| if x > 0.9999 { 1.0 } else { x })
}

fn model_parameters(p: &ParameterSet) {
    let conf = config::base_config(None);
    let model_config = config::get_config::<RSNNModel<BaseModel>>();
    let (m1, m2) =  BaseModel::parse_params(p, &model_config);

    let t_cpm = normalize(m1.mapv(|x| math::ml::sigmoid(x)));
    let input_t_cpm = normalize(m2.mapv(|x| math::ml::sigmoid(x)));

    println!("t_cpm:\n{}", t_cpm);
    println!("input_t_cpm:\n{}", input_t_cpm);
}

fn analyze<T: Task + TaskEval>(stats: OptimizationStatistics) {
    let mut best: Option<(f32, &DefaultRepresentation, &ParameterSet)> = None;
    for (i, r) in stats.runs.iter().enumerate() {
        let run_best = r.best();
        print!("[{i}] fitness: {} ", run_best.0);

        accuracy::<T>(run_best.1);
        if let Some((f, _, _)) = best {
            if run_best.0 > f {
                best= Some(run_best);
            }
        } else {
            best = Some(run_best);
        }
    }

    println!("best fitness: {}", best.unwrap().0);

    analyze_run::<T>(best.unwrap().1);

    //analyze_model(best.unwrap().2);
    //accuracy::<T>(best.unwrap().1)
    //analysis::analyze_network(best.unwrap().1);
}

fn parse_args() -> Option<String> {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        return Some(args[1].clone());
    }

    None
}

fn report_stats(stats: &OptimizationStatistics) {
    for r in &stats.runs {
        println!("gens: {}", r.generations.len());
    }

}

fn load_stats(path: &str) -> OptimizationStatistics {
    match data::load::<OptimizationStatistics>(path) {
        Ok(r) => { r },
        Err(e) =>     { println!("error: {e}"); std::process::exit(1); }
    }
}

fn load_experiment_report(path: &str) -> ExperimentReport {
    match data::load::<ExperimentReport>(path) {
        Ok(r) => { r },
        Err(e) =>     { println!("error: {e}"); std::process::exit(1); }
    }
}


fn main() {
    logger::init_logger(Some("debug".to_string()));

    let path: String = match parse_args() {
        Some(a) => {a},
        None => {println!("Add arguments"); std::process::exit(0) }
    };


    //let stats = load_stats(&path);
    let report = load_experiment_report(&path);
    let stats = report.stats;
    println!("# runs: {}", stats.runs.len());

    plot_individual_runs(&stats);
    //analyze::<PatternSimilarityTask>(stats);


    //analyze::<MultiPatternTask>(stats);
}
