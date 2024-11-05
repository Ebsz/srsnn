use crate::config::BaseConfig;

use serde::{Serialize, Deserialize};

use evolution::stats::OptimizationStatistics;


#[derive(Clone, Deserialize, Serialize)]
pub struct ExperimentReport {
    pub stats: OptimizationStatistics,
    pub conf: BaseConfig,
    pub version: String
}
