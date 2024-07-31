use luna::process::Process;
use luna::process::{optimize, hyper};
use luna::config::{base_config, BaseConfig};

use utils::random;
use utils::logger::init_logger;

use std::env;


fn parse_config_name_from_args() -> Option<String> {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        return Some(args[1].clone());
    }

    None
}

fn run_process(config: BaseConfig) {
    match config.process.as_str() {
        "optimize" => { optimize::OptimizationProcess::init(config); },
        "hyper"    => { hyper::HyperOptimization::init(config); },
        _          => { println!("Unknown process: {}", config.process); }
    }
}

fn main() {
    let config_name = parse_config_name_from_args();
    let config = base_config(config_name.clone());

    init_logger(config.log_level.clone());
    log::debug!("Using config: {}", config_name.unwrap_or("default".to_string()));
    log::debug!("seed is {}", random::SEED);

    run_process(config);
}
