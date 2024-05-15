use serde::Deserialize;


#[derive(Clone, Copy, Debug, Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub parent_fraction: f32,
    pub elites: usize,

    // Stop conditions
    pub fitness_goal: f32,
    pub max_generations: u32,

    pub n_stale_before_reset: usize
}
