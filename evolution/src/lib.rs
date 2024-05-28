pub mod genome;
pub mod config;
pub mod population;
pub mod stats;

use genome::Genome;


pub trait Evaluate<G, P> {
    fn eval(&self, g: &G) -> (f32, P);
}

#[derive(Debug, Clone)]
pub struct EvolutionEnvironment {
    pub inputs: usize,
    pub outputs: usize,
}
