use model::Model;
use model::network::representation::DefaultRepresentation;

use tasks::Task;

use evolution::Evaluate;
use evolution::algorithm::Algorithm;

use utils::environment::Environment;
use crate::eval::Evaluation;


//use std::time::Instant;




pub fn optimize<M: Model, T: Task, A: Algorithm<M>>(
    conf: MainConf<M, A>,
    mut eval: impl Evaluate<M, DefaultRepresentation>,
    env: Environment)
{
    //log::info!("Algorithm: {}", A::name());

    let mut algo = A::new(conf.algorithm, &conf.model);

    let mut gen = 0;
    loop {
        let ps = algo.parameter_sets();
        // Create models from parameter sets
        let mut models = vec![];

        for p in ps {
            let m = M::new(&conf.model, p, &env);
            models.push(m);
        }

        let mut e = vec![];
        for (i, m) in models.iter().enumerate() {
            e.push((i as u32, m));
        }

        let evaluations = eval.eval(&e);
        let fitness: Vec<f32> = evaluations.iter().map(|e| e.1).collect();

        log_generation(gen, &sorted_fitness(evaluations));

        algo.step(fitness);

        gen += 1
    }
}

// TODO: Rename to something else
pub struct MainConf<M: Model, A: Algorithm<M>> {
    pub model: M::Config,
    pub algorithm: A::Config
}

fn sorted_fitness(evals: Vec<Evaluation>) -> Vec<(u32, f32)> {
    let mut sorted_fitness: Vec<(u32, f32)> = evals.iter()
        .map(|g| (g.0, g.1)).collect();

    sorted_fitness.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    assert!(sorted_fitness.windows(2).all(|f| f[0].1 >= f[1].1));

    sorted_fitness
}

fn log_generation(gen: usize, sorted_fitness: &Vec<(u32, f32)>) {
    let mean_fitness: f32 = sorted_fitness.iter().map(|(_, f)| f).sum::<f32>() / sorted_fitness.len() as f32;
    let best_fitness: f32 = sorted_fitness[0].1;

    log::info!("Generation {} - best fit: {}, mean: {}",
        gen, best_fitness, mean_fitness);

    //stats.log_generation(best_fitness, mean_fitness);
}
