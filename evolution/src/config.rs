use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub parent_fraction: f32,
    pub elites: usize,

    // Stop conditions
    pub fitness_goal: f32,
    pub max_generations: u32,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        EvolutionConfig {
            population_size: 50,
            parent_fraction: 0.3,
            elites: 2,
            fitness_goal: 90.0,
            max_generations: 50,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct GenomeConfig {
    pub max_neurons: usize,

    pub initial_neuron_count_range: (usize, usize),
    pub initial_connection_count_range: (usize, usize),

    pub n_mutations: usize,

    pub mutate_connection_probability: f32,
    pub mutate_toggle_connection_probability: f32,
    pub mutate_add_connection_probability: f32,
    pub mutate_add_neuron_probability: f32,
}

impl Default for GenomeConfig {
    fn default() -> Self {
        GenomeConfig {
            max_neurons: 50,
            initial_neuron_count_range: (2, 5),
            initial_connection_count_range: (3, 4),
            n_mutations: 2,
            mutate_connection_probability: 0.8,
            mutate_toggle_connection_probability: 0.3,
            mutate_add_connection_probability: 0.03,
            mutate_add_neuron_probability: 0.02
        }
    }
}
