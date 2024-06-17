use serde::de::DeserializeOwned;

use core::fmt::Debug;


pub trait Configurable {
    type Config: ConfigSection;
}

pub trait ConfigSection: DeserializeOwned + Debug {
    fn name() -> String;
}
