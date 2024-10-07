pub mod algorithm;
pub mod stats;


#[derive(Debug, Clone)]
pub struct EvolutionEnvironment {
    pub inputs: usize,
    pub outputs: usize,
}

pub trait Evaluate<G, P> {
    fn eval(&mut self, g: &[(u32, &G)]) -> Vec<(u32, f32, P)>;
}
