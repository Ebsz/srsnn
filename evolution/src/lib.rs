//! Evolutionary algorithm that evolves a spiking network.
//!
//! After initializing, the population is evolved by
//!     1. Calculating fitness
//!     2. Select best-fit genomes for reproduction
//!     3. Create new population from crossover
//!     4. Mutate new population
//!
//! This is repeated until for a set number of generations,
//! or until a genome is found that has a fitness > FITNESS_GOAL.
//!

pub mod genome;
pub mod config;
pub mod population;

use genome::Genome;


pub type Fitness = fn(&Genome, &EvolutionEnvironment) -> f32;

#[derive(Debug, Clone)]
pub struct EvolutionEnvironment {
    pub fitness: Fitness,
    pub inputs: usize,
    pub outputs: usize,
}
