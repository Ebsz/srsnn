const DEFAULT_LOG_LEVEL: &'static str = "trace";


pub fn init_logger(custom_level: Option<String>) {
    let mut level: String = DEFAULT_LOG_LEVEL.to_string();

    if let Some(l) = custom_level {
        level = l;
    }

    let env = env_logger::Env::default()
        .filter_or("RUST_LOG", level)
        .write_style("always");

    env_logger::Builder::from_env(env)
        .format_timestamp_millis()
        .init();
}
