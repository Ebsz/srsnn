use serde::de::DeserializeOwned;


pub trait Configurable {
    type Config: ConfigSection;
}

pub trait ConfigSection: DeserializeOwned {
    fn name() -> String;
}
