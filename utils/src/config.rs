use serde::de::DeserializeOwned;

use core::fmt::Debug;


pub trait Configurable {
    type Config: ConfigSection + Sync;
}

pub trait ConfigSection: DeserializeOwned + Debug + Clone {
    fn name() -> String;
}
