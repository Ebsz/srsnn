//! Base debugging executable

use utils::logger::init_logger;

mod network;
mod single_neuron;

const LOG_LEVEL: &str = "debug";

fn main() {
    init_logger(Some(LOG_LEVEL.to_string()));

    //network::init();

    single_neuron::init();
}
