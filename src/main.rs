use srsnn::process::Process;
use srsnn::process::{default, hyper, experiment, test};
use srsnn::config::{base_config, BaseConfig};

use utils::random;
use utils::logger::init_logger;

use std::env;


fn run_process(config: BaseConfig) {
    match config.process.as_str() {
        "default"    => { default::DefaultProcess::init(config); },
        "hyper"      => { hyper::HyperOptimization::init(config); },
        "experiment" => { experiment::Experiment::init(config); },
        "test"       => { test::TestProcess::init(config); },
        _            => { println!("Unknown process: {}", config.process); }
    }
}

fn parse_config_name_from_args() -> Option<String> {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        return Some(args[1].clone());
    }

    None
}

fn main() {
    let config_name = parse_config_name_from_args();
    let config = base_config(config_name.clone());

    init_logger(config.log_level.clone());
    log::debug!("Using config: {}", config_name.unwrap_or("default".to_string()));

    random::random_seed();

    run_process(config);
}
