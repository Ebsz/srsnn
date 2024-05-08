pub mod matrix_genome;
pub mod pool_genome;

use crate::EvolutionEnvironment;

use utils::config::ConfigSection;


pub trait Genome {
    type Config: ConfigSection;

    fn new(env: &EvolutionEnvironment, config: &Self::Config) -> Self;

    fn mutate(&mut self, config: &Self::Config);
    fn crossover(&self, other: &Self) -> Self;
}
