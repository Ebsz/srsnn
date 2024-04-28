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
