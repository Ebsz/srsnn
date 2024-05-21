pub mod genome;
pub mod config;
pub mod population;
pub mod stats;

use genome::Genome;


pub trait Evaluate<G> {
    fn eval(&self, g: &G) -> f32;
}

#[derive(Debug, Clone)]
pub struct EvolutionEnvironment {
    pub inputs: usize,
    pub outputs: usize,
}
