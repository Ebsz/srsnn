//pub mod nes;
pub mod snes;

use utils::config::Configurable;
use utils::parameters::ParameterSet;


pub trait Algorithm: Configurable {
    fn new(conf: Self::Config, params: ParameterSet) -> Self;

    fn step(&mut self, evals: Vec<f32>);
    fn parameter_sets(&self) -> &[ParameterSet];
}
