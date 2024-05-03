use serde::de::DeserializeOwned;

pub trait ConfigSection: DeserializeOwned {
    fn name() -> String;
}
