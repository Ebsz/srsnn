pub mod config;

pub mod uniform;
pub mod typed_uniform;
pub mod er;

pub mod udd_base; // Uniform dual dynamics
pub mod utd_base; // Uniform Typed Dynamics
pub mod edd_base; // Evolved dual dynamics

pub mod base;
pub mod ed;

pub mod ex1_ablation;

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

pub mod blk {
    use super::*;

    pub trait GeneratorComponent<X> {
        type Parameters;

        fn get(p: Self::Parameters) -> X;
    }

    pub mod weights {
        use super::*;

        use rand::Rng;

        fn default() -> ValueSet {
            ValueSet { f: Arc::new(
                move |_i, _j| 1.0
            )}
        }

        fn uniform(max: f32) -> ValueSet {
            ValueSet { f: Arc::new(
                move |_i, _j| rand::thread_rng().gen_range(0.0..max)
            )}
        }
    }

    pub mod dynamics {
        use super::*;

        use model::neuron::izhikevich::IzhikevichParameters;
        use rand::Rng;

        use ndarray::{Array, Array2};

        //pub type DynamicsComponent = GeneratorComponent<NeuronSet>;

        fn default() -> NeuronSet {
            NeuronSet { f: Arc::new(
                move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
            )}
        }

        pub fn uniform() -> NeuronSet {
            let r = IzhikevichParameters::RANGES;

            NeuronSet {f: Arc::new(
                move |_i| array![
                    rand::thread_rng().gen_range(r[0].0..r[0].1),     // a
                    rand::thread_rng().gen_range(r[1].0..r[1].1),     // b
                    rand::thread_rng().gen_range(r[2].0..r[2].1),     // c
                    rand::thread_rng().gen_range(r[3].0..r[3].1),     // d
                    if rand::random::<f32>() > 0.5 { 1.0 } else { 0.0 }
                ]
            )}
        }

        /// Returns a neuron set that maps from k types to uniformly generated dynamical parameters
        pub fn uniform_typed(k: usize) -> NeuronSet {
            let pr = IzhikevichParameters::RANGES;

            let mut d: Array2<f32> = Array::zeros((k, 5));

            for mut row in d.rows_mut() {
                row[0] = rand::thread_rng().gen_range(pr[0].0..pr[0].1);
                row[1] = rand::thread_rng().gen_range(pr[1].0..pr[1].1);
                row[2] = rand::thread_rng().gen_range(pr[2].0..pr[2].1);
                row[3] = rand::thread_rng().gen_range(pr[3].0..pr[3].1);
                row[4] = if rand::random::<f32>() > 0.5 { 1.0 } else { 0.0 };
            }

            NeuronSet::from_value(d)
        }
    }
}
