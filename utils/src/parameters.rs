use ndarray::{s, Array, Array1, Array2};

use serde::{Serialize, Deserialize};


pub trait Parameterized {
    fn params() -> ParameterSet;
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub enum Parameter {
    Scalar(f32),
    Vector(Array1<f32>),
    Matrix(Array2<f32>)
}

impl Parameter {
    pub fn len(&self) -> usize {
        match self {
            Parameter::Scalar(_) => { 1 },
            Parameter::Vector(v) => { v.len() }
            Parameter::Matrix(m) => { m.len() }
        }
    }

    pub fn linearize(&self) -> Vec<f32> {
        match self {
            Parameter::Scalar(s) => { vec![*s] },
            Parameter::Vector(v) => { v.to_vec() }
            Parameter::Matrix(m) => { m.clone().into_raw_vec() }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ParameterSet {
    pub set: Vec<Parameter>
}

impl ParameterSet {
    pub fn size(&self) -> usize {
        self.set.iter().map(|p| p.len()).sum()
    }

    /// Create a parameter set with the same structure from a linear representation
    pub fn assign(&self, data: &Array1<f32>) -> ParameterSet {
        assert!(data.len() == self.size());

        let mut ps: Vec<Parameter> = vec![];

        let mut a = 0;
        for p in &self.set {
            let d = data.slice(s![a..(a + p.len())]);

            a += p.len();

            match p {
                Parameter::Scalar(_) => { ps.push(Parameter::Scalar(d[0])); },
                Parameter::Vector(_) => { ps.push(Parameter::Vector(d.into_owned())); },
                Parameter::Matrix(m) => {
                    let a = m.shape()[0];
                    let b = m.shape()[1];

                    ps.push(Parameter::Matrix(d.into_shape((a,b)).unwrap().into_owned()));
                }
            }
        }

        ParameterSet {
            set: ps
        }
    }

    /// Convert the parameter set to a linear representation
    pub fn linearize(&self) -> Array1<f32> {
        let v: Vec<f32> = self.set.iter().map(|p| p.linearize()).flatten().collect();

        Array::from_vec(v)
    }

    pub fn is_nan(&self) -> bool {
        self.set.iter().all(|a| a.linearize().iter().all(|x| x.is_nan()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_linearize() {
        let a = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        let m = Parameter::Matrix(Array::zeros((2, 2)));
        let p = Parameter::Vector(Array::zeros(2));

        let mut p = ParameterSet { set: vec![m, p] };
        p.assign(&a);

        let lin = p.linearize();
        let re = p.assign(&lin);

        assert!(p == re);
    }
}
