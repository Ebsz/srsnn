pub mod base;
pub mod ed;

use crate::models::generator_model::ModelConfig;

use csa::{ConnectionSet, ValueSet, NeuronSet, NetworkSet};
use csa::mask::Mask;

use utils::config::Configurable;
use utils::parameters::ParameterSet;
use utils::environment::Environment;
use utils::random;

use ndarray::array;

use std::sync::Arc;
use std::fmt::Debug;


pub trait Generator: Configurable + Clone + Debug + Sync {
    fn get(p: &ParameterSet, config: &ModelConfig<Self>, env: &Environment) -> (NetworkSet, ConnectionSet);
    fn params(config: &ModelConfig<Self>, env: &Environment) -> ParameterSet;

    fn default_dynamics() -> NeuronSet {
        NeuronSet { f: Arc::new(
            move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        )}
    }

    fn random_weights(min: f32, max: f32) -> ValueSet {
        ValueSet { f: Arc::new(
            move |_i, _j| random::random_range((min, max))
        )}
    }

    fn default_weights() -> ValueSet {
        ValueSet { f: Arc::new(
            move |_i, _j| 1.0
        )}
    }

    fn default_output() -> Mask {
        Mask { f: Arc::new( move |i, j| i == j ) }
    }

    /// 4 Izhikevich parameters + inhibitory flag
    const N_DYNAMICAL_PARAMETERS: usize = 5;
}

pub struct NetworkModel {
    pub c: ConnectionSet,
    pub d: NeuronSet,
    pub i: ConnectionSet
}

mod components {

}
