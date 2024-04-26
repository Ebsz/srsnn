use evolution::config::{EvolutionConfig, GenomeConfig};

use tasks::TaskName;

use config::{Config, ConfigError, File};
use serde::Deserialize;

const DEFAULT_CONFIG_PATH: &str = "config/default.toml";

#[derive(Debug, Deserialize)]
pub struct MainConfig {
    pub task: String,
    pub evolution: EvolutionConfig,
    pub genome: GenomeConfig,
}


impl MainConfig {
    fn new(config_path: &str) -> Result<Self, ConfigError> {
        let mut config = Config::builder()
            .add_source(File::with_name(DEFAULT_CONFIG_PATH))
            .add_source(File::with_name(config_path))
            .build()
            .unwrap();

        config.try_deserialize()
    }
}

pub fn get_config(conf_path: Option<&str>) -> MainConfig {
    let mut path = DEFAULT_CONFIG_PATH;

    if let Some(p) = conf_path {
        path = p;
    }

    MainConfig::new(path).unwrap()
}


pub fn get_taskname(task_str: &String) -> TaskName {
    match task_str.as_str() {
        "SurvivalTask" => TaskName::SurvivalTask,
        "CatchingTask" => TaskName::CatchingTask,
        "MovementTask" => TaskName::MovementTask,
        _ => {panic!("Unknown task {}", task_str);}
    }
}
