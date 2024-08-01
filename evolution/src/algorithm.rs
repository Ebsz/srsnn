pub mod nes;
pub mod snes;

use utils::config::Configurable;
use utils::parameters::ParameterSet;

use model::Model;


pub trait Algorithm: Configurable {
    fn new<M: Model>(conf: Self::Config, model_conf: &M::Config) -> Self;

    fn step(&mut self, evals: Vec<f32>);
    fn parameter_sets(&self) -> &[ParameterSet];
}
