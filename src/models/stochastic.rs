//! Stochastic models of RSNNs

pub mod base_model;
pub mod random_model;

use crate::models::Model;

use utils::config::ConfigSection;

use serde::Deserialize;


pub trait StochasticModel: Model {

}

#[derive(Deserialize)]
pub struct StochasticGenomeConfig {
    pub max_neurons: usize
}

impl ConfigSection for StochasticGenomeConfig {
    fn name() -> String {
        "stochastic_genome".to_string()
    }
}
