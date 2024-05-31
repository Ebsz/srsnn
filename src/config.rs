use evolution::config::EvolutionConfig;
use evolution::genome::Genome;

use utils::config::ConfigSection;

use config::{Config, ConfigError, File};
use serde::Deserialize;

use std::cell::RefCell;


const CONFIG_DIR: &str = "config/";
const DEFAULT_CONFIG_PATH: &str = "config/default.toml";

thread_local! {
    static CONFIG: RefCell<Option<Config>> = RefCell::new(None);
}

#[derive(Debug, Deserialize)]
pub struct MainConfig {
    pub task: String,
    pub model: String,
    pub log_level: Option<String>,
    pub evolution: EvolutionConfig
}

impl MainConfig {
    fn new(config: Config) -> Result<Self, ConfigError> {
        config.try_deserialize()
    }
}

fn read_config(config_path: Option<String>) -> Result<Config, ConfigError>{
    let mut builder = Config::builder()
        .add_source(File::with_name(DEFAULT_CONFIG_PATH));


    if let Some(path) = config_path {
        builder = builder.add_source(File::with_name(path.as_str()));
    }

    Ok(builder.build()?)
}

pub fn get_config(config_name: Option<String>) -> MainConfig {
    let mut path: Option<String> = None;

    // Parse config path from name
    if let Some(name) = config_name {
        path = Some([CONFIG_DIR, name.as_str(), ".toml"].join(""));
    }

    // Read the config file.
    match read_config(path.clone()) {
        Ok(config) => {
            CONFIG.replace(Some(config));
        },
        Err(e) => {
            println!("Could not read config: {e}");

            std::process::exit(-1);
        }
    }

    // Deserialize into MainConfig.
    let config = CONFIG.with(|c| c.borrow().clone()).unwrap();

    match MainConfig::new(config) {
        Ok(config) => { config },
        Err(e) => {
            println!("Could not load config: {e}");

            std::process::exit(-1);
        }
    }
}

pub fn genome_config<G: Genome>() -> G::Config {
    let config = CONFIG.with(|c| c.borrow().clone()).unwrap();

    match config.get::<G::Config>(G::Config::name().as_str()) {
        Ok(c) => c,
        Err(e) => {
            println!("Error loading genome config: {e}");

            std::process::exit(-1);
        }
    }
}
