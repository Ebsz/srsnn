//! Stochastic models of RSNNs

pub mod base_model;
pub mod random_model;

use utils::config::ConfigSection;

use serde::Deserialize;

#[derive(Deserialize)]
pub struct StochasticGenomeConfig {
    pub max_neurons: usize
}

impl ConfigSection for StochasticGenomeConfig {
    fn name() -> String {
        "stochastic_genome".to_string()
    }
}
