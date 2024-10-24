use ndarray::{Array, Array1};

use std::fmt;


#[derive(Clone, Debug)]
pub struct Spikes {
    pub data: Array1<bool>
}

impl Spikes {
    pub fn new(n: usize) -> Spikes {
        Spikes {
            data: Array::zeros(n).mapv(|_: f32| false)
        }
    }

    // TODO: implement and use AsRef instead; requires making data private
    // and only giving access through refs, so we can keep track of changes
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

impl Into<Array1<u32>> for &Spikes {
    fn into(self) -> Array1<u32> {
        self.data.map(|s| if *s { 1 } else { 0})
    }
}

impl From<Array1<u32>> for Spikes {
    fn from(a: Array1<u32>) -> Self {
        Spikes {
            // NOTE: Defining this as != 0 means that values > 1 are ignored
            data: a.mapv(|v| if v != 0 { true } else { false })
        }
    }
}

impl fmt::Display for Spikes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d: Array1<u32> = self.into();

        write!(f, "{}", d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    #[test]
    fn test_spikes_firing_correct() {
        const N: usize = 10;

        let mut s = Spikes::new(N);

        assert!(s.firing().len() == 0);

        s.data.slice_mut(s![0..2]).fill(true);

        let f = s.firing();

        assert!(f.contains(&0));
        assert!(f.contains(&1));
        assert!(f.len() == 2);
    }

    #[test]
    fn test_spikes_as_float() {
        const N: usize = 2;

        let s = Spikes::new(N);

        let a: Array1<f32> = Array::zeros(N);
        assert_eq!(s.as_float(), a);
    }
}
