use serde::Deserialize;

use utils::config::{Configurable, ConfigSection};


#[derive(Debug, Deserialize)]
pub struct TrialConfig {
    pub trials: usize,
}

impl ConfigSection for TrialConfig {
    fn name() -> String {
        "eval".to_string()
    }
}

#[derive(Debug, Deserialize)]
pub struct BatchConfig {
    pub batch_size: usize
}

impl ConfigSection for BatchConfig {
    fn name() -> String {
        "batch".to_string()
    }
}

pub struct Batch;
impl Configurable for Batch {
    type Config = BatchConfig;
}
