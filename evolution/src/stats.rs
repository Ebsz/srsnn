use model::network::representation::DefaultRepresentation;

use utils::math;
use utils::parameters::ParameterSet;

use serde::{Serialize, Deserialize};

// Best, mean, stddev
type Generation = (f32, f32, f32);


#[derive(Clone, Deserialize, Serialize)]
pub struct Run {
    pub generations: Vec<Generation>,

    pub best_network: Option<(f32, DefaultRepresentation, ParameterSet)>
}
impl Run {
    fn new() -> Self {
        Run {
            generations: Vec::new(),
            best_network: None,
        }
    }

    fn log(&mut self, best: f32, mean: f32, std: f32, network: (DefaultRepresentation, ParameterSet)) {
        self.generations.push((best, mean, std));

        if let Some((f, _, _)) = self.best_network {
            if best > f {
                self.best_network = Some((best, network.0, network.1));
            }
        } else {
            self.best_network = Some((best, network.0, network.1));
        }
    }

    /// Current generation number
    fn gen(&self) -> usize {
        self.generations.len()
    }

    // Return the best individual of the run
    fn best(&self) -> (f32, &DefaultRepresentation, &ParameterSet) {
        match &self.best_network {
            Some((f, r, p)) => { (*f, &r, &p) },
            None => { panic!("best called on Run with no best network"); }
        }
    }

    pub fn best_series(&self) -> Vec<f32> {
        self.generations.iter().map(|g| g.0).collect()
    }

    pub fn mean_series(&self) -> Vec<f32> {
        self.generations.iter().map(|g| g.1).collect()
    }

    pub fn stddev_series(&self) -> Vec<f32> {
        self.generations.iter().map(|g| g.2).collect()
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

    pub fn empty() -> OptimizationStatistics {
        OptimizationStatistics {
            runs: vec![]
        }
    }

    pub fn log_generation(&mut self, best: f32, mean: f32, std: f32,
        best_network: (DefaultRepresentation, ParameterSet)) {
        self.runs.last_mut().unwrap().log(best, mean, std, best_network);
    }


    pub fn new_run(&mut self) {
        self.runs.push(Run::new());
    }

    pub fn push_run(&mut self, run: Run) {
        self.runs.push(run);
    }

    pub fn run(&self) -> &Run {
        self.runs.last().unwrap()
    }

    /// Total number of generations
    pub fn sum_generations(&self) -> usize {
        self.runs.iter().map(|r| r.generations.len()).sum()
    }

    //pub fn best_fit(&self) -> Vec<f32> {
    //    self.runs.iter().fold(vec![], |acc, x| [acc, x.best_fitness.clone()].concat())
    //}

    pub fn best(&self) -> (f32, &DefaultRepresentation, &ParameterSet) {
        self.runs.iter().map(|r| r.best()).max_by(|a,b| a.0.partial_cmp(&b.0).expect("")).unwrap()
    }
}
