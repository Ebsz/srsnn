const LOG_LEVEL: &'static str = "info";


pub fn init_logger() {

    let env = env_logger::Env::default()
        .filter_or("RUST_LOG", LOG_LEVEL)
        .write_style("always");

    env_logger::Builder::from_env(env)
        .format_timestamp_millis()
        .init();
}
