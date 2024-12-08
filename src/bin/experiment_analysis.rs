/// Analysis and plotting of experiment reports.

use srsnn::plots;
use srsnn::config;
use srsnn::eval;

use srsnn::models::generator_model::GeneratorModel;
use srsnn::models::generator::base::BaseModel;

use srsnn::process::experiment::report::ExperimentReport;


use model::network::representation::DefaultRepresentation;
use evolution::stats::OptimizationStatistics;

use tasks::{Task, TaskEval};
use tasks::pattern::PatternTask;
use tasks::multipattern::MultiPatternTask;
use tasks::pattern_similarity::PatternSimilarityTask;
use tasks::time_series::TimeSeriesTask;
use tasks::time_series::ts::SinSeries;

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


/// Evaluates a network over a number of setups, returning the fitness and, optionally,  accuracy
fn evaluate<T: Task + TaskEval> (repr: &DefaultRepresentation, setups: &[T::Setup]) -> (f32, Option<f32>) {
    let results = eval::run_network_on_task::<T>(repr, setups);

    let accuracy = T::accuracy(&results);
    let val = T::fitness(results);

    (val, accuracy)

}

fn normalize(d: Array2<f32>) -> Array2<f32> {
    d
        .mapv(|x| if x < 0.0001 { 0.0 } else { x })
        .mapv(|x| if x > 0.9999 { 1.0 } else { x })
}

fn base_model_parameters(p: &ParameterSet) {
    let conf = config::base_config(None);
    let model_config = config::get_config::<GeneratorModel<BaseModel>>();
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

        let validation_setups = T::eval_setups();
        let (fitness, accuracy) = evaluate::<T>(run_best.1, &validation_setups);
        match accuracy {
            Some(acc) =>  {
                println!("validation fitness: {:.3}, accuracy: {:.3}", fitness, acc);
            },
            None => { println!("validation fitness: {:.3}", fitness); }
        }
        if let Some((f, _, _)) = best {
            if run_best.0 > f {
                best= Some(run_best);
            }
        } else {
            best = Some(run_best);
        }
    }

    println!("best fitness: {}", best.unwrap().0);

//    analyze_run::<T>(best.unwrap().1);

    //analyze_model(best.unwrap().2);
    //accuracy::<T>(best.unwrap().1)
    //analysis::analyze_network(best.unwrap().1);
}

fn report_stats(stats: &OptimizationStatistics) {
    for r in &stats.runs {
        println!("gens: {}", r.generations.len());
    }
}

fn multiple_reports<T: Task + TaskEval>(reports: Vec<ExperimentReport>) {
    log::info!("{} reports", reports.len());

    log::info!("v: {}", reports[0].version);
    for r in reports {
        let mut best = single_report::<T>(r);

        // sort by acc
        //best.sort_by(|x,y| y.2.partial_cmp(&x.2).unwrap());

        //for b in best {
        //    println!("{:.3}\t{:.3}\t{:.3}\t{}", b.0, b.1, b.2, b.3);
        //}
    }
}

fn single_report<T: Task + TaskEval>(report: ExperimentReport) -> Vec<(f32, f32, Option<f32>, usize)> {
    let desc = report.desc.unwrap();
    let n_runs = report.stats.runs.len();

    log::info!("report: {}", desc);
    log::info!("{} runs", n_runs);

    let best = ana::n_best_runs(&report.stats, 10);
    //let fitness: Vec<f32> = best.iter().map(|x| x.0).collect();

    let mut stats: Vec<(f32, f32, Option<f32>, usize)> = vec![];

    let validation_setups = T::eval_setups();

    let mut report_best: Option<(f32, DefaultRepresentation)> = None;

    println!("n \t f\te_f\te_acc\truns");
    for (i, b) in best.iter().enumerate() {
        let (fitness, accuracy) = evaluate::<T>(&b.0.1, &validation_setups);

        if let Some(a) = accuracy {
            println!("{i}\t\t{:.3}\t\t{:.3}\t\t{:.3}\t\t{}", b.0.0, fitness, a, b.1);
        } else {
            println!("{i}\t\t{:.3}\t\t{:.3}\t\t{:.3}\t\t{}", b.0.0, fitness, "-", b.1);
        }

        //model_parameters(&b.0.2);

        if let Some(c) = &report_best {
            if fitness > c.0 {
                report_best = Some((fitness, b.0.1.clone()));
            }
        } else {
            report_best = Some((fitness, b.0.1.clone()));
        }

        stats.push((b.0.0, fitness, accuracy, b.1));
    }

    let (best_eval, best_repr) = report_best.unwrap();
    save_network(&best_repr, format!("{}_eval_{}.json", desc, best_eval).as_str());

    stats
}

fn save_network(r: &DefaultRepresentation, path: &str) {
    let res = data::save::<DefaultRepresentation>(r.clone(), path);

    match res {
        Ok(_) => (println!("Network saved to {}", path)),
        Err(e) => println!("Error saving network: {:?}", e),
    }
}

fn parse_args() -> Option<Vec<String>> {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        return Some(args[1..].into());
    }

    None
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

    let args: Vec<String> = match parse_args() {
        Some(a) => {a},
        None => {println!("Add arguments"); std::process::exit(0) }
    };

    log::info!("loading experiment reports");
    let reports: Vec<ExperimentReport> = args.iter().map(|p| load_experiment_report(p)).collect();

    //single_report::<PatternSimilarityTask>(reports[0].clone());
    multiple_reports::<TimeSeriesTask<SinSeries>>(reports);

    //multiple_reports(reports);

    //let stats = load_stats(&path);
    //let report = load_experiment_report(&path[0]);
    //let stats = report.stats;
    //println!("# runs: {}", stats.runs.len());

    ////plot_individual_runs(&stats);
    ////analyze::<PatternSimilarityTask>(stats);

    //match report.conf.task.as_str() {
    //    "pattern"               => { analyze::<PatternTask>(stats); },
    //    "pattern_similarity"    => { analyze::<PatternSimilarityTask>(stats); },
    //    "multipattern"          => { analyze::<MultiPatternTask>(stats); },
    //    t => { println!("Unknown task: {t}"); }
    //}

    //analyze::<MultiPatternTask>(stats);
}

//fn plot_5_best_runs() {
//
//}

mod ana {
    use super::*;

    pub fn n_best_runs(s: &OptimizationStatistics, n: usize) -> Vec<((f32, DefaultRepresentation, ParameterSet), usize)> {
        let mut best: Vec<((f32, DefaultRepresentation, ParameterSet), usize)> =
            s.runs.iter().map(|r| (r.best_network.clone().unwrap(),r.generations.len()) ).collect();

        best.sort_by(|x,y| y.0.0.partial_cmp(&x.0.0).unwrap());

        best[..n].to_vec()
    }
}

mod etc {
    use super::*;
    pub fn update_report_description(args: Vec<String>) {
        if args.len() != 2 {
            println!("Expected two args: [path] [desc]");
            std::process::exit(0);
        }

        let path = &args[0];
        let desc = &args[1];

        update_desc(path, desc);
    }

    fn update_desc(path: &str, desc: &String) {
        log::info!("Loading report {}", path);

        let mut report = load_experiment_report(path);

        log::info!("n runs: {}", report.stats.runs.len());
        log::info!("Old description: '{:?}'", report.desc);

        log::info!("Adding description: '{desc}'");
        report.desc = Some(desc.clone());

        data::save::<ExperimentReport>(report, path);
        log::info!("Saved updated report to {path}");
    }
}
