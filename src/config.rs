use utils::config::{Configurable, ConfigSection};

use config::{Config, ConfigError, File};
use serde::Deserialize;

use std::cell::RefCell;


const CONFIG_DIR: &str = "config/";
const DEFAULT_CONFIG_PATH: &str = "config/default.toml";

thread_local! {
    static CONFIG: RefCell<Option<Config>> = RefCell::new(None);
}



#[derive(Debug, Deserialize)]
pub struct BaseConfig {
    pub process: String,
    pub task: String,
    pub model: String,
    pub log_level: Option<String>,
}

impl BaseConfig {
    fn new(config: Config) -> Result<Self, ConfigError> {
        config.try_deserialize()
    }
}

fn read_config(path: Option<String>) -> Result<Config, ConfigError>{
    let mut builder = Config::builder()
        .add_source(File::with_name(DEFAULT_CONFIG_PATH));

    if let Some(p) = path {
        builder = builder.add_source(File::with_name(p.as_str()));
    }

    Ok(builder.build()?)
}

pub fn base_config(config_name: Option<String>) -> BaseConfig {
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

    // Deserialize into BaseConfig.
    let config = CONFIG.with(|c| c.borrow().clone()).unwrap();
    match BaseConfig::new(config) {
        Ok(config) => { config },
        Err(e) => {
            println!("Could not load config: {e}");

            std::process::exit(-1);
        }
    }
}

pub fn get_config<C: Configurable>() -> C::Config {
    let config = CONFIG.with(|c| c.borrow().clone()).unwrap();

    match config.get::<C::Config>(C::Config::name().as_str()) {
        Ok(c) => c,
        Err(e) => {
            println!("Error loading config: {e}");

            std::process::exit(-1);
        }
    }
}
