pub mod genome;
pub mod config;
pub mod population;
pub mod stats;


pub trait Evaluate<G, P> {
    fn eval(&self, g: &G) -> (f32, P);

    /// Called at the end of the eval
    fn next(&mut self);
}

#[derive(Debug, Clone)]
pub struct EvolutionEnvironment {
    pub inputs: usize,
    pub outputs: usize,
}
