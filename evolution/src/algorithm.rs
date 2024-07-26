pub mod nes;

use utils::config::{ConfigSection, Configurable};
use utils::parameters::ParameterSet;

use model::Model;


pub trait Algorithm<M: Model>: Configurable {
    fn new(conf: Self::Config, model_conf: &M::Config) -> Self;

    fn step(&mut self, evals: Vec<f32>);
    fn parameter_sets(&self) -> &[ParameterSet];
}
