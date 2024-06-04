//! Stochastic models of RSNNs

pub mod main_model;

use crate::models::Model;

use utils::config::ConfigSection;

use serde::Deserialize;

use model::network::representation::{DefaultRepresentation};


pub trait StochasticModel: Model {
    fn sample(&self) -> DefaultRepresentation;
}

impl<S: StochasticModel> Model for S {
    fn develop(&self) -> DefaultRepresentation {
        self.sample()
    }
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
