use evolution::config::{EvolutionConfig, GenomeConfig};
use tasks::config::TaskConfig;

use tasks::TaskName;

use config::{Config, ConfigError, File};
use serde::Deserialize;

const DEFAULT_CONFIG_PATH: &str = "config/default.toml";

#[derive(Debug, Deserialize)]
pub struct MainConfig {
    pub task: TaskConfig,
    pub evolution: EvolutionConfig,
    pub genome: GenomeConfig,
}

impl MainConfig {
    fn new(config_path: Option<String>) -> Result<Self, ConfigError> {
        let mut builder = Config::builder()
            .add_source(File::with_name(DEFAULT_CONFIG_PATH));


        if let Some(path) = config_path {
            builder = builder.add_source(File::with_name(path.as_str()));
        }

        let config = builder.build()?;

        config.try_deserialize()
    }
}

pub fn get_config(config_path: Option<String>) -> MainConfig {
    match MainConfig::new(config_path.clone()) {
        Ok(config) => {
            log::debug!("Using {}", config_path.unwrap_or("default config".to_string()));

            config
        }
        Err(e) => {
            println!("Could not load config: {e}");

            std::process::exit(-1);
        }
    }
}


pub fn get_taskname(task_str: &String) -> TaskName {
    match task_str.as_str() {
        "SurvivalTask" => TaskName::SurvivalTask,
        "CatchingTask" => TaskName::CatchingTask,
        "MovementTask" => TaskName::MovementTask,
        _ => {panic!("Unknown task {}", task_str);}
    }
}
