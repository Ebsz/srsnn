//! Test process.

use crate::config::BaseConfig;
use crate::process::{Process, MainConf};

use crate::analysis::{graph, run_analysis};
use crate::plots;

use crate::eval;

use model::Model;
use model::network::representation::DefaultRepresentation;

use tasks::{Task, TaskEval, TaskInput};
use tasks::task_runner::{TaskRunner, Runnable};

use tasks::testing::{TestTask, TestTaskSetup};
use tasks::multipattern::{MultiPatternTask, MultiPatternSetup};
use tasks::time_series::TimeSeriesTask;
use tasks::time_series::ts::{LorenzSystem, SinSeries};

use evolution::algorithm::Algorithm;
use evolution::algorithm::snes::SeparableNES;
use evolution::stats::OptimizationStatistics;

use utils::math;
use utils::analysis;
use utils::parameters::ParameterSet;
use utils::environment::Environment;

use ndarray::{s, Array, Array1, Array2};


pub struct TestProcess;
impl Process for TestProcess {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        log::info!("Running test process");

        //test_time_series_task();
        //test_multipattern_task();
        //Self::pattern_test(conf);

        //let main_conf = Self::main_conf::<RSNNModel<TestModel>, TimeSeriesTask<SinSeries>, SeparableNES>();
        //let env = Self::environment::<TimeSeriesTask<SinSeries>>();
        //Self::log_config(&conf, &main_conf, &env);

        //let p = RSNNModel::<TestModel>::params(&main_conf.model, &env);

        // Get parameter set from SNES
        //log::info!("Getting parameter sets");
        //let snes = SeparableNES::new(main_conf.algorithm.clone(), p);
        //let params = snes.parameter_sets();

        // Create and develop model from the first parameter set
        //log::info!("Developing model");
        //let model = RSNNModel::<TestModel>::new(&main_conf.model, &params[0], &env);
        //let r = model.develop();

        let path = "out/network_ed_model_sin_time_series_919830.json";      // dynamics model
        let r: DefaultRepresentation = utils::data::load(path).unwrap();
        println!("{:#?}", r.env);

        let analysis = crate::analysis::analyze_network(&r);

        let setups = TimeSeriesTask::<SinSeries>::eval_setups();

        let record = run_analysis::<TimeSeriesTask<SinSeries>>(&r, &setups)[0].clone();

        let results: Vec<<TimeSeriesTask<SinSeries> as Task>::Result>
            = eval::run_network_on_task::<TimeSeriesTask<SinSeries>>(&r, &setups);

        let fitness = TimeSeriesTask::<SinSeries>::fitness(results);
        println!("fitness: {fitness}");

    }
}

fn test_time_series_task() {
    let setups: Vec<_> = TimeSeriesTask::<SinSeries>::eval_setups();

    let mut results = vec![];

    let mut output: Vec<Array1<f32>> = vec![];
    let mut input: Vec<Array1<f32>> = vec![];

    for i in 0..setups.len() {
        let mut task = TimeSeriesTask::<SinSeries>::new(&setups[i]);

        let mut t = 0;
        loop {
            let inp: Array1<f32> = if t > 50 && t < 90 {
                utils::encoding::rate_encode(&(Array::<f32, _>::ones(10) * 0.3))
            } else if t > 90 {
                utils::encoding::rate_encode(&(Array::<f32, _>::ones(10) * 0.1))
            } else {
                Array::zeros(10)
            };

            let data: Vec<u32> = inp.iter().enumerate()
                .filter_map(|(i, x)| if *x == 1.0 { Some(i as u32) } else { None }).collect();

            input.push(inp);

            let s = task.tick(TaskInput{ data });

            if let Some(r) = s.result {
                results.push(r);
                break;
            }

            output.push(s.output.data);

            t+=1;
        }
    }

    //let accuracy = TimeSeriesTask::accuracy(&results).unwrap();
    //let fitness = TimeSeriesTask::fitness(results);
    //log::info!("fitness:  {fitness}, accuracy: {accuracy}");

    let _ = plots::plot_spikes(output, "task_output.png");
    let _ = plots::plot_spikes(input, "task_input.png");
}

fn test_multipattern_task() {
    let setups: Vec<_> = MultiPatternTask::eval_setups();

    let mut results = vec![];

    let mut output: Vec<Array1<f32>> = vec![];
    let mut input: Vec<Array1<f32>> = vec![];


    for i in 0..setups.len() {
        let mut task = MultiPatternTask::new(&setups[i]);

        let mut t = 0;
        loop {
            let inp: Array1<f32> = if t > 50 && t < 90 {
                utils::encoding::rate_encode(&(Array::<f32, _>::ones(10) * 0.3))
            } else if t > 90 {
                utils::encoding::rate_encode(&(Array::<f32, _>::ones(10) * 0.1))
            } else {
                Array::zeros(10)
            };

            let data: Vec<u32> = inp.iter().enumerate()
                .filter_map(|(i, x)| if *x == 1.0 { Some(i as u32) } else { None }).collect();

            input.push(inp);

            let s = task.tick(TaskInput{ data });

            if let Some(r) = s.result {
                results.push(r);
                break;
            }

            output.push(s.output.data);

            t+=1;
        }
    }

    let accuracy = MultiPatternTask::accuracy(&results).unwrap();
    let fitness = MultiPatternTask::fitness(results);
    log::info!("fitness:  {fitness}, accuracy: {accuracy}");

    let _ = plots::plot_spikes(output, "task_output.png");
    let _ = plots::plot_spikes(input, "task_input.png");
}


fn network_sample_analysis<M: Model>(model: M) -> (f32, f32) {
    let n = 100;

    let mut density = Array::zeros(n);
    //let mut input_density = Vec::new();

    for i in 0..n {
        let network = model.develop();

        let graph: graph::Graph = (&network).into();
        let d = graph::GraphAnalysis::density(&graph);

        density[i] = d;

        //let n_input_connections: u32 = r.input_cm.iter().sum();
        //let input_density = n_input_connections as f32 / (r.input_cm.shape()[0] * r.input_cm.shape()[1]) as f32;
    }

    (density.mean().unwrap(), density.std(1.0))
}
