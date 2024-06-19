use serde::Serialize;
use serde::de::DeserializeOwned;

use std::fs;
use std::fs::File;
use std::io::prelude::*;


const DEFAULT_OUT_DIR: &str = "out";

pub fn save<T: Serialize>(x: T, path: &str) -> std::io::Result<()> {
    let serialized_string = serde_json::to_string(&x).unwrap();

    fs::create_dir_all(DEFAULT_OUT_DIR);

    write_file(serialized_string, path)?;

    Ok(())
}

pub fn load<T: DeserializeOwned>(path: &str) -> std::io::Result<T> {
    let data = read_file(path)?;

    let d: T = serde_json::from_str(data.as_str())?;

    Ok(d)
}

fn write_file(data: String, path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    file.write_all(data.as_bytes());

    Ok(())
}

pub fn read_file(path: &str) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut data = String::new();

    file.read_to_string(&mut data)?;

    Ok(data)
}
