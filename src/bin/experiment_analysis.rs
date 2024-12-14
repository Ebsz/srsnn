/// Analysis and plotting of experiment reports.

use srsnn::plots;
use srsnn::config;
use srsnn::eval;
use srsnn::config::BaseConfig;

use srsnn::models::generator_model::GeneratorModel;
use srsnn::models::generator::base::BaseModel;

use srsnn::process::experiment::report::ExperimentReport;

use model::network::representation::DefaultRepresentation;
use evolution::stats::{OptimizationStatistics, Run};

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

use serde::{Serialize, Deserialize};
use ndarray::{Array, Array1, Array2};

use std::env;

use datamodel::*;


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
                best = Some(run_best);
            }
        } else {
            best = Some(run_best);
        }
    }

    println!("best fitness: {}", best.unwrap().0);
}

fn analyze_report<T: Task + TaskEval>(report: ExperimentReport, validation_setups: &[T::Setup]) -> ReportAnalysis {
    let desc = report.desc.clone().unwrap();
    let n_runs = report.stats.runs.len();

    log::info!("report: {}", desc);
    log::info!("{} runs", n_runs);

    let mut analysis = ReportAnalysis {
        conf: report.conf,
        version: report.version,
        desc: report.desc,
        runs: vec![]
    };


    for (i, r) in report.stats.runs.iter().enumerate() {
        if let Some((fitness, network, ps)) = &r.best_network {
            let (val, accuracy) = evaluate::<T>(&network, validation_setups);

            let run = RunAnalysis {
                run: r.clone(),
                best_network: NetworkEval {
                    network: network.clone(),
                    fitness: *fitness,
                    validation: val,
                    accuracy,

                    runs: r.generations.len(),
                },
                model_params: ps.clone()
            };

            analysis.runs.push(run);

        } else {
            println!("Cannot evaluate run with no best network");
        }
    }

    analysis.runs.sort_by(|x,y| y.best_network.validation.partial_cmp(&x.best_network.validation).expect(""));
    analysis
}

fn single_report<T: Task + TaskEval>(report: ExperimentReport) -> Vec<NetworkEval> {
    let desc = report.desc.unwrap();
    let n_runs = report.stats.runs.len();

    log::info!("report: {}", desc);
    log::info!("{} runs", n_runs);

    let runs: Vec<((f32, DefaultRepresentation, ParameterSet), usize)>
         = alys::n_best_runs(&report.stats, 1000);

    //let mut stats: Vec<(f32, f32, Option<f32>, usize)> = vec![];
    let mut evals: Vec<NetworkEval> = vec![];

    let validation_setups = T::eval_setups();

    log::info!("{} validation setups", validation_setups.len());

    for (i, b) in runs.iter().enumerate() {
        let (val, accuracy) = evaluate::<T>(&b.0.1, &validation_setups);
        //model_parameters(&b.0.2);

        evals.push( NetworkEval {
            network: b.0.1.clone(),
            fitness: b.0.0,
            validation: val,

            accuracy: None,

            runs: b.1

        });
    }

    evals.sort_by(|x,y| y.validation.partial_cmp(&x.validation).expect(""));

    let best: &NetworkEval = &evals[0];
    //println!("{}", best.fitness);

    let (best_eval, best_repr) = (best.validation, best.network.clone());

    // Run best network
    let record = &srsnn::analysis::run_analysis::<T>(&best_repr, &validation_setups)[0];

    let name = format!("{}_eval_{}", desc, best_eval);
    srsnn::plots::plot_run_spikes(&record, Some((name + ".png").as_str()));

    //save_network(&best_repr, (name + ".json").as_str());

    evals
}


fn multiple_reports<T: Task + TaskEval>(reports: Vec<ExperimentReport>) {
    let mut experiment_analysis = ExperimentAnalysis { reports: vec![] };

    log::info!("{} reports", reports.len());

    log::info!("v: {}", reports[0].version);
    let validation_setups = T::eval_setups();

    for r in reports {
        //let mut best = single_report::<T>(r);
        let report_analysis: ReportAnalysis = analyze_report::<T>(r, &validation_setups);

        println!("n \t\t f\t\tval\t\te_acc\t\truns");
        for (i, ra) in report_analysis.runs[..25].iter().enumerate() {
            let b = &ra.best_network;

            println!("{i}\t\t{:.3}\t\t{:.3}\t\t{:.3}\t\t{}", b.fitness, b.validation, "-", b.runs);
        }

        experiment_analysis.reports.push(report_analysis);
    }

    // save experiment analysis
    let path = "analysis.json";
    let res = data::save::<ExperimentAnalysis>(experiment_analysis, path);
    match res {
        Ok(_) => println!("Saved to {}", path),
        Err(e) => println!("Error: {:?}", e),
    }
}

fn save_network(r: &DefaultRepresentation, path: &str) {
    let p = "out/networks/".to_string() + path;

    let res = data::save::<DefaultRepresentation>(r.clone(), p.as_str());

    match res {
        Ok(_) => (println!("Network saved to {}", p)),
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



fn get_avg_eval(mut analysis: ExperimentAnalysis) {
    // Number of runs we calculate the average over.
    let n_runs = 50;

    // Sort
    for mut report in analysis.reports {
        report.runs.sort_by(|x,y| y.best_network.validation.partial_cmp(&x.best_network.validation).expect(""));

        println!("{}:", report.desc.unwrap());

        let mean: f32 = report.runs[..n_runs].iter()
            .map(|r| r.best_network.validation).sum::<f32>() / n_runs as f32;
        println!("mean of top {n_runs} : {:.3}:", mean);

        println!("------");
    }
}

fn main() {
    logger::init_logger(Some("debug".to_string()));

    let args: Vec<String> = match parse_args() {
        Some(a) => {a},
        None => {println!("Add arguments"); std::process::exit(0) }
    };
    //log::info!("loading experiment reports");
    //let reports: Vec<ExperimentReport> = args.iter().map(|p| load_experiment_report(p)).collect();
    //log::info!("loaded {} reports", &reports.len());

    //multiple_reports::<TimeSeriesTask<SinSeries>>(reports);

    log::info!("Loading ExperimentAnalysis");
    let mut analysis: ExperimentAnalysis = alys::load_analysis(&args[0]);

    //get_avg_eval(analysis);

    //alys::analyze_best_networks(analysis);
    alys::print_analysis(analysis);

    //analyze_best_networks(analysis);
    //get_avg_eval(analysis);

}

mod etc {
    use super::*;

    use serde::Serialize;

    pub fn save_object<O: Clone + Serialize>(o: &O, path: &str) {
        let res = data::save::<O>(o.clone(), path);

        match res {
            Ok(_) => (println!("Object saved to {}", path)),
            Err(e) => println!("Error saving : {:?}", e),
        }
    }
}


mod alys {
    use super::*;

    pub fn load_analysis(path: &str) -> ExperimentAnalysis {
        match data::load::<ExperimentAnalysis>(path) {
            Ok(r) => { r },
            Err(e) =>     { println!("error: {e}"); std::process::exit(1); }
        }
    }

    pub fn print_analysis(mut analysis: ExperimentAnalysis) {
        println!("version: {}", analysis.reports[0].version);
        println!("task: {}", analysis.reports[0].conf.task);

        for mut report in analysis.reports {
            report.runs.sort_by(|x,y| y.best_network.validation.partial_cmp(&x.best_network.validation).expect(""));
            if let Some(d) = report.desc {
                println!("\n{d} ({} runs)", report.runs.len());
            } else {
                println!("---------");
            }

            println!("n \t\t f\t\tval\t\te_acc\t\tgenerations");
            for (i, ra) in report.runs[..25].iter().enumerate() {
                let b = &ra.best_network;

                println!("{i}\t\t{:.3}\t\t{:.3}\t\t{:.3}\t\t{}", b.fitness, b.validation, "-", b.runs);
            }
        }
    }

    pub fn analyze_best_networks(mut analysis: ExperimentAnalysis) {
        // Sort
        for mut report in analysis.reports {
            // Sort the runs by validation fitness, descending
            report.runs.sort_by(|x,y| y.best_network.validation.partial_cmp(&x.best_network.validation).expect(""));

            let desc = report.desc.unwrap();
            println!("{}:", desc);

            let val = report.runs[0].best_network.validation;
            println!("val: {:.3}:", val);

            let repr = &report.runs[0].best_network.network;
            let (graph, graph_analysis) = srsnn::analysis::analyze_network(repr);

            let validation_setups = vec![TimeSeriesTask::<SinSeries>::eval_setups()[0].clone()];
            let record = &srsnn::analysis::run_analysis::<TimeSeriesTask<SinSeries>>(&repr, &validation_setups)[0];

            let name = format!("{}_eval_{}", desc, val);
            srsnn::plots::plot_run_spikes(&record, Some((name.clone() + ".png").as_str()));
            save_network(&repr, (name + ".json").as_str());

            println!("------");
        }
    }

    pub fn save_mean_of_n_best(analysis: ExperimentAnalysis, n: usize) {
        let out_dir = "out/means/";
        println!("finding mean of {n} best networks and saving to {out_dir}");

        for mut report in analysis.reports {
            report.runs
                .sort_by(|x,y| y.best_network.validation.partial_cmp(&x.best_network.validation).expect(""));

            let runs: Vec<Run> = report.runs[..n].iter().map(|x| x.run.clone()).collect();

            let evals: Vec<(f32, f32, f32)> = alys::mean_eval(&runs);

            let best: Array1<f32> = evals.iter().map(|e| e.0).collect();
            let mean: Array1<f32> = evals.iter().map(|e| e.1).collect();
            let stddev: Array1<f32> = evals.iter().map(|e| e.2).collect();

            let desc = &report.desc.as_ref().unwrap();

            etc::save_object::<Array1<f32>>(&best,
                format!("{}avg_evals_best_{}.json", out_dir, desc).as_str());
            etc::save_object::<Array1<f32>>(&mean,
                format!("{}avg_evals_mean_{}.json", out_dir, desc).as_str());
            etc::save_object::<Array1<f32>>(&stddev,
                format!("{}avg_evals_stddev_{}.json", out_dir, desc).as_str());

        }
    }

    pub fn n_best_runs(s: &OptimizationStatistics, n: usize)
        -> Vec<((f32, DefaultRepresentation, ParameterSet), usize)> {

        let mut best: Vec<((f32, DefaultRepresentation, ParameterSet), usize)> =
            s.runs.iter().filter_map(|r|
                if let Some(network) = r.best_network.clone() {
                    Some((network, r.generations.len()))
                } else {
                    None
                }).collect();

        best.sort_by(|x,y| y.0.0.partial_cmp(&x.0.0).unwrap());

        let i = best.len().min(n);

        best[..i].to_vec()
    }

    /// Returns the mean (best, mean, stddev) across a set of runs.
    pub fn mean_eval(runs: &[Run]) -> Vec<(f32, f32, f32)> {
        let n_runs = runs.len();

        let max_g = runs.iter().map(|r| r.generations.len()).max().unwrap();
        let min_g = runs.iter().map(|r| r.generations.len()).min().unwrap();
        println!("max: {max_g} generations, min: {min_g} generations");

        let mut evals = vec![];

        for g in 0..max_g {

            let vals: Vec<(f32,f32,f32)> = runs.iter().filter_map(|r| if let Some(gen) = r.generations.get(g) { Some(*gen) } else { None }).collect();

            let vlen = vals.len();


            let b: (f32, f32, f32) = (vals.iter().map(|x| x.0).sum::<f32>() / vlen as f32,
                                      vals.iter().map(|x| x.1).sum::<f32>() / vlen as f32,
                                      vals.iter().map(|x| x.2).sum::<f32>() / vlen as f32);


            evals.push(b);
        }

        evals
    }
}

mod report {
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

mod datamodel {
    use super::*;

    #[derive(Clone, Deserialize, Serialize)]
    pub struct ExperimentAnalysis {
        pub reports: Vec<ReportAnalysis>,
    }

    // Contains the analysis of a single ExperimentReport
    #[derive(Clone, Deserialize, Serialize)]
    pub struct ReportAnalysis {
        pub conf: BaseConfig,
        pub version: String,
        pub desc: Option<String>,
        pub runs: Vec<RunAnalysis>
    }

    #[derive(Clone, Deserialize, Serialize)]
    pub struct RunAnalysis {
        pub run: Run,
        pub model_params: ParameterSet,
        pub best_network: NetworkEval
    }

    #[derive(Clone, Deserialize, Serialize)]
    pub struct NetworkEval {
        pub network: DefaultRepresentation,

        pub fitness:  f32,      // Evaluation fitness
        pub validation: f32,    // Validation fitness

        pub accuracy: Option<f32>,
        pub runs: usize  // Number of runs
    }

}
