pub mod genome;
pub mod config;
pub mod population;
pub mod stats;
pub mod evolvable;


pub trait Evaluate<G, P> {
    fn eval(&mut self, g: &[(u32, &G)]) -> Vec<(u32, f32, P)>;
}

#[derive(Debug, Clone)]
pub struct EvolutionEnvironment {
    pub inputs: usize,
    pub outputs: usize,
}
