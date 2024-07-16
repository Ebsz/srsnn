use crate::EvolutionEnvironment;

use utils::config::Configurable;


pub trait Genome: Configurable {
    fn new(env: &EvolutionEnvironment, config: &Self::Config) -> Self;

    fn mutate(&mut self, config: &Self::Config);

    /// This is called on the genome with greater or equal fitness
    fn crossover(&self, other: &Self) -> Self;
}
