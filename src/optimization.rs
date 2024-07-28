use crate::eval::Evaluation;
use crate::analysis::{Graph, GraphAnalysis};

use model::Model;
use model::network::representation::DefaultRepresentation;

use tasks::Task;

use evolution::Evaluate;
use evolution::algorithm::Algorithm;

use utils::environment::Environment;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};


pub fn optimize<M: Model, T: Task, A: Algorithm<M>>(
    conf: MainConf<M, A>,
    mut eval: impl Evaluate<M, DefaultRepresentation>,
    env: Environment)
{
    let mut algo = A::new(conf.algorithm, &conf.model);

    let stop_signal = Arc::new(AtomicBool::new(false));
    init_ctrl_c_handler(stop_signal.clone());

    let mut gen = 0;
    while !stop_signal.load(Ordering::SeqCst) {
        let ps = algo.parameter_sets();
        // Create models from parameter sets
        let mut models = vec![];

        for p in ps {
            let m = M::new(&conf.model, p, &env);
            models.push(m);
        }

        let e: Vec<(u32, &M)> = models.iter().enumerate()
            .map(|(i, m)| (i as u32, m)).collect();

        let evaluations = eval.eval(&e);
        let fitness: Vec<f32> = evaluations.iter().map(|e| e.1).collect();

        log_generation(gen, &evaluations);

        algo.step(fitness);

        gen += 1
    }
}

fn init_ctrl_c_handler(stop_signal: Arc<AtomicBool>) {
    let mut stopped = false;

    ctrlc::set_handler(move || {
        if stopped {
            std::process::exit(1);
        } else {
            log::info!("Stopping..");

            stopped = true;
            stop_signal.store(true, Ordering::SeqCst);
        }
    }).expect("Error setting Ctrl-C handler");

    log::info!("Use Ctrl-C to stop gracefully");
}

// TODO: Rename to something else
pub struct MainConf<M: Model, A: Algorithm<M>> {
    pub model: M::Config,
    pub algorithm: A::Config
}

fn sorted_fitness(evals: &[Evaluation]) -> Vec<(u32, f32)> {
    let mut sorted_fitness: Vec<(u32, f32)> = evals.iter()
        .map(|g| (g.0, g.1)).collect();

    sorted_fitness.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    assert!(sorted_fitness.windows(2).all(|f| f[0].1 >= f[1].1));

    sorted_fitness
}

fn log_generation(gen: usize, evals: &[Evaluation]) {
    let sorted = sorted_fitness(evals);

    let mean_fitness: f32 = sorted.iter().map(|(_, f)| f).sum::<f32>() / sorted.len() as f32;
    let best_fitness: f32 = sorted[0].1;

    let best: &DefaultRepresentation = evals.iter()
        .filter_map(|(i,_,r)| if *i == sorted[0].0 {Some(r)} else {None})
        .collect::<Vec<&DefaultRepresentation>>()[0];

    analyze_model(best);

    log::info!("Generation {} - best fit: {}, mean: {}",
        gen, best_fitness, mean_fitness);

    //stats.log_generation(best_fitness, mean_fitness);
}

fn analyze_model(r: &DefaultRepresentation) {
    let g: Graph = r.into();

    let ga = GraphAnalysis::analyze(&g);

    log::info!("Best graph: {}\n{}", g, ga);
}
