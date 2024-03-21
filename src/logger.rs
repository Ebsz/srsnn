const DEFAULT_LOG_LEVEL: &'static str = "trace";

pub fn init_logger() {
    use std::io::Write;

    let env = env_logger::Env::default()
        .filter_or("RUST_LOG", DEFAULT_LOG_LEVEL)
        .write_style("always");

    env_logger::Builder::from_env(env)
        .format_timestamp_millis()
        .init();
}
