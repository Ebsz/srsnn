//! Abstract representation of an object's environment, defining its ports

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    pub inputs: usize,
    pub outputs: usize,
}
