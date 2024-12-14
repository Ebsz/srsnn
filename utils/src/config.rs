use serde::de::DeserializeOwned;

use core::fmt::Debug;


pub trait Configurable {
    type Config: ConfigSection + Sync;
}

pub trait ConfigSection: DeserializeOwned + Debug + Clone {
    fn name() -> String;
}

use serde::Deserialize;
#[derive(Clone, Debug, Deserialize)]
pub struct EmptyConfig {

}

impl ConfigSection for EmptyConfig {
    fn name() -> String {
        "NOP".to_string()
    }
}
