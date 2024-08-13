use model::network::representation::DefaultRepresentation;

use utils::math;

use serde::{Serialize, Deserialize};


#[derive(Deserialize, Serialize)]
pub struct Run {
    pub generations: usize,
    pub mean_fitness: Vec<f32>,
    pub best_fitness: Vec<f32>,
    pub best_network: Vec<DefaultRepresentation>
}
impl Run {
    fn new() -> Self {
        Run {
            generations: 0,
            mean_fitness: Vec::new(),
            best_fitness: Vec::new(),
            best_network: Vec::new(),
        }
    }

    fn log(&mut self, best: f32, mean: f32, r: DefaultRepresentation) {
        self.mean_fitness.push(mean);
        self.best_fitness.push(best);
        self.best_network.push(r);

        self.generations += 1;
    }

    /// Current generation number
    fn gen(&self) -> usize {
        self.generations
    }

    // Return the best individual of the run
    fn best(&self) -> (f32, &DefaultRepresentation) {
        let best_ix = math::max_index(&self.best_fitness);

        (self.best_fitness[best_ix], &self.best_network[best_ix])

        //self.best_fitness.iter().zip(best_network.iter()).

    }
}

#[derive(Deserialize, Serialize)]
pub struct OptimizationStatistics {
    pub runs: Vec<Run>
}

impl OptimizationStatistics {
    pub fn new() -> OptimizationStatistics {
        OptimizationStatistics {
            runs: vec![Run::new()]
        }
    }

    pub fn log_generation(&mut self, best: f32, mean: f32, r: DefaultRepresentation) {
        self.runs.last_mut().unwrap().log(best, mean, r);
    }

    pub fn new_run(&mut self) {
        self.runs.push(Run::new());
    }

    pub fn run(&self) -> &Run {
        self.runs.last().unwrap()
    }

    /// Total number of generations
    pub fn sum_generations(&self) -> usize {
        self.runs.iter().map(|r| r.generations).sum()
    }

    pub fn best_fit(&self) -> Vec<f32> {
        self.runs.iter().fold(vec![], |acc, x| [acc, x.best_fitness.clone()].concat())
    }

    pub fn best(&self) -> (f32, &DefaultRepresentation) {
        self.runs.iter().map(|r| r.best()).max_by(|a,b| a.0.partial_cmp(&b.0).expect("")).unwrap()
    }
}
