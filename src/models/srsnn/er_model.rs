//! Model based on the Erdős-Rényi random graph, which generates SRSNNs similar to its namesake.

use crate::csa;
use crate::csa::DynamicsSet;
use crate::csa::mask::Mask;

use crate::models::srsnn::SRSNN;

use utils::config::{ConfigSection, Configurable};
use serde::Deserialize;
use std::sync::Arc;

use ndarray::array;


pub struct ERModel {
    pub p: f32
}

impl Configurable for ERModel {
    type Config = ERModelConfig;
}

impl SRSNN for ERModel {
    fn new(c: Self::Config) -> Self {
        ERModel {
            p: c.p
        }
    }

    fn connectivity(&self) -> Mask {
        csa::mask::random(self.p)
    }

    fn dynamics(&self) -> DynamicsSet {
        DynamicsSet { f: Arc::new(
            move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        )}
    }
}

#[derive(Debug, Deserialize)]
pub struct ERModelConfig {
    pub p: f32
}

impl ConfigSection for ERModelConfig {
    fn name() -> String {
        "er_model".to_string()
    }
}
