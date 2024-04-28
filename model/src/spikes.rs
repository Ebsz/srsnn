use ndarray::{Array, Array1};


#[derive(Debug)]
pub struct Spikes {
    pub data: Array1<bool>
}

impl Spikes {
    pub fn new(n: usize) -> Spikes {
        Spikes {
            data: Array::zeros(n).mapv(|_: f32| false)
        }
    }

    pub fn as_float(&self) -> Array1<f32> {
        self.data.mapv(|x| if x { 1.0 } else { 0.0 })

    }

    /// Get the indices of neurons that fire
    pub fn firing(&self) -> Vec<usize> {
        self.data.iter().enumerate().filter(|(_, n)| **n).map(|(i,_)| i).collect()
    }

    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }
}
