use serde::Deserialize;
use utils::config::ConfigSection;


#[derive(Clone, Debug, Deserialize)]
pub struct TypedConfig {
    pub k: usize,       // # of types

    pub k_in: usize,    // # of input types
    pub k_out: usize,   // # of output types
}

impl ConfigSection for TypedConfig {
    fn name() -> String {
        "typed".to_string()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct GeometricConfig {
    pub distance_threshold: f32,
    pub max_coordinate: f32,
}

impl ConfigSection for GeometricConfig {
    fn name() -> String {
        "geometric".to_string()
    }
}
