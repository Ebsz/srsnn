use model::network::representation::DefaultRepresentation;

pub struct EvolutionStatistics {
    pub generation_mean_fitness: Vec<f32>,
    pub generation_best_fitness: Vec<f32>,
    pub generation_best_network: Vec<DefaultRepresentation>
}

impl EvolutionStatistics {
    pub fn new() -> EvolutionStatistics {
        EvolutionStatistics {
            generation_mean_fitness: Vec::new(),
            generation_best_fitness: Vec::new(),
            generation_best_network: Vec::new(),
        }
    }

    pub fn log_generation(&mut self, best: f32, mean: f32, r: DefaultRepresentation) {
        self.generation_mean_fitness.push(mean);
        self.generation_best_fitness.push(best);
        self.generation_best_network.push(r);
    }
}
