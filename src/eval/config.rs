use serde::Deserialize;

use utils::config::ConfigSection;

#[derive(Deserialize)]
pub struct TrialConfig {
    pub trials: usize,
}

impl ConfigSection for TrialConfig {
    fn name() -> String {
        "eval".to_string()
    }
}

#[derive(Deserialize)]
pub struct BatchSetup {
    pub batch_size: usize,
    pub batch_index: usize,
}
