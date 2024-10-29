use ndarray::{Array, Array1};

use std::fmt;

use num_traits::Num;


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

    /// Get the indices of neurons that fire
    pub fn firing(&self) -> Vec<usize> {
        self.data.iter().enumerate().filter(|(_, n)| **n).map(|(i,_)| i).collect()
    }

    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }
}

impl<T: Num> Into<Array1<T>> for &Spikes {
    fn into(self) -> Array1<T> {
        self.data.map(|s| if *s { T::one() } else { T::zero() })
    }
}

impl From<Array1<u32>> for Spikes {
    fn from(a: Array1<u32>) -> Self {
        assert!(a.iter().all(|x| *x == 0 || *x == 1));

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
        assert_eq!(s.into(), a);
    }
}
