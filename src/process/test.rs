//! Default process.
//!
//! Performs a single optimization run.

use crate::config::BaseConfig;
use crate::process::{Process, MainConf};
use crate::eval::MultiEvaluator;
use crate::optimization::Optimizer;

use crate::analysis::{graph, run_analysis};

use crate::runnable::RunnableNetwork;
use crate::models::rsnn::{RSNN, RSNNModel};
use crate::models::srsnn::gt_model::GeometricTypedModel;
use crate::models::srsnn::test::TestModel;
use crate::plots;
use crate::plots::plt;
use crate::eval;

use model::Model;
use model::DefaultNetwork;

use tasks::{Task, TaskEval, TaskInput};
use tasks::testing::{TestTask, TestTaskSetup};
use tasks::task_runner::{TaskRunner, Runnable};

use evolution::algorithm::Algorithm;
use evolution::algorithm::snes::SeparableNES;
use evolution::stats::OptimizationStatistics;

use utils::math;
use utils::analysis;
use utils::parameters::ParameterSet;
use utils::environment::Environment;

use ndarray::{s, Array, Array1};


pub struct TestProcess;
impl Process for TestProcess {
    fn run<M: Model, T: Task + TaskEval>(conf: BaseConfig) {
        log::info!("Running test process");

        let main_conf = Self::main_conf::<RSNNModel<TestModel>, TestTask, SeparableNES>();
        let env = Self::environment::<TestTask>();
        Self::log_config(&conf, &main_conf, &env);

        let p = RSNNModel::<TestModel>::params(&main_conf.model, &env);

        // Get parameter set from SNES
        log::info!("Getting parameter sets");
        let snes = SeparableNES::new(main_conf.algorithm.clone(), p);
        let params = snes.parameter_sets();

        // Create and develop model from the first parameter set
        log::info!("Developing model");
        let model = RSNNModel::<TestModel>::new(&main_conf.model, &params[0], &env);
        let r = model.develop();

        println!("{:#?}", r.env);

        let record = run_analysis::<TestTask>(&r);
        plots::plot_run_spikes(&record);
        plots::plot_all_potentials(&record);

        log::info!("Running network on task");
        let setups: Vec<_> = TestTask::eval_setups();
        let results: Vec< <TestTask as Task>::Result > = eval::run_network_on_task::<TestTask>(&r, &setups);
        log::info!("got results");

        let record = results[0].record.clone();
        let fr = analysis::firing_rate(results[0].record.clone(), 10);

        for i in 0..fr.shape()[0] {
            println!("{} \t\t {}", fr.slice(s![i, ..]), record.slice(s![i, ..]));
        }

        //let fitness = TestTask::fitness(results);
        //log::info!("fitness:  {fitness}");

        let plot_ok = plt::plot_matrix(&fr, "firing_rate.png");
        //let plot_ok = plt::plot_matrix(&normalized_fr, "nfiring_rate.png");

        //match plot_ok {
        //    Ok(_) => (),
        //    Err(e) => println!("Error creating plot: {:?}", e),
        //}

        //let c = visual::base::BaseComponent::<RSNNModel<TestModel>>::new(&main_conf.model, params, &env);
        //visual::window(vec![
        //    Box::new(c),
        //]);

        //for i in 0..10 {
        //    let model = RSNNModel::<GeometricTypedModel>::new(&main_conf.model, &params[i], &env);

        //    for j in 0..10 {
        //        let r = model.develop();

        //        let d = r.network_cm.map(|x| *x as f32);

        //        let plot_ok = plt::plot_matrix(&d, format!("out/plots/connectivity_{}_{}.png", i, j).as_str());

        //        match plot_ok {
        //            Ok(_) => (),
        //            Err(e) => println!("Error creating plot: {:?}", e),
        //        }
        //    }

        //}

        //let mut runnable = RunnableNetwork::<DefaultNetwork>::build(&network);

        //let task = TestTask::new(&TestTaskSetup { });
        //let mut runner = TaskRunner::new(task, &mut runnable);

        //log::info!("running");
        //let result = runner.run();
    }
}

fn test_task() {
    let mut task = TestTask::new(&TestTaskSetup { });

    let mut results = vec![];

    let mut output = vec![];

    let mut t = 0;
    loop {
        let data: Vec<u32>  = (0..10).map(|i| i).collect();
        let s = task.tick(TaskInput{ data });

        if let Some(r) = s.result {
            results.push(r);
            break;
        }

        output.push(s);

        t+=1;
    }
}

//fn model_analysis<M: Model>(ps: &[ParameterSet], model_conf: &M::Config, env: &Environment) {
//    log::info!("Performing model analysis");
//    log::info!("# parameter sets: {}", ps.len());
//
//    let mut means: Array1<f32> = Array::zeros(100);
//    let mut stds: Array1<f32> = Array::zeros(100);
//
//    for i in 0..100 {
//        let model = M::new(model_conf, &ps[i], env);
//
//        let (mean, std) = network_sample_analysis(model);
//
//        means[i] = mean;
//        stds[i] = std;
//    }
//
//    println!("means: {:#?}", means);
//
//    println!("max: {}, min: {}",
//        math::maxf(means.as_slice().unwrap()),
//        math::minf(means.as_slice().unwrap()));
//
//    println!("{:#?}", stds);
//    println!("max: {}, min: {}",
//        math::maxf(stds.as_slice().unwrap()),
//        math::minf(stds.as_slice().unwrap()));
//}


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
        //.push(graph_analysis.density);
    }

    (density.mean().unwrap(), density.std(1.0))

    //println!("{:?}", density);
    //println!("mean: {:?}", density.mean().unwrap());
    //println!("std: {:?}", density.std(1.0));
}
