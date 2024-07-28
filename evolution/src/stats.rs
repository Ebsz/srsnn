use model::network::representation::DefaultRepresentation;

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

}

pub struct EvolutionStatistics {
    pub runs: Vec<Run>
}

impl EvolutionStatistics {
    pub fn new() -> EvolutionStatistics {
        EvolutionStatistics {
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

    pub fn sum_generations(&self) -> usize {
        self.runs.iter().map(|r| r.generations).sum()
    }
}
