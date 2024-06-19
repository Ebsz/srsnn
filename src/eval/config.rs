use serde::Deserialize;

use utils::config::{Configurable, ConfigSection};


#[derive(Debug, Deserialize)]
pub struct EvalConfig {
    pub max_threads: usize,
    pub trials: usize,
}

impl ConfigSection for EvalConfig {
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
