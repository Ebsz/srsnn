//! Generic containers for different data structures
//! with functions for performing mutation and crossover

use utils::random;

use ndarray::{s, Array1, Array, Array2};
use ndarray_rand::rand_distr::Uniform;


//trait EvolvableParameter {
//}
//
//pub struct VectorParameter<T> {
//    data: Array1<T>
//}
//
//impl<T> EvolvableParameter for VectorParameter<T> { }
//
//pub struct MatrixParameter<T> {
//    data: Array2<T>
//}
//
//impl<T> EvolvableParameter for MatrixParameter<T> { }



/// Generic matrix representation for arbitrary kinds of data,
#[derive(Clone, Debug)]
pub struct MatrixGene {
    pub data: Array2<f32>,
}

impl MatrixGene {
    pub fn init_random(n: usize, range: (f32, f32)) -> MatrixGene {
        MatrixGene {
            data: random::random_matrix((n, n), Uniform::new(range.0, range.1))
        }
    }

    /// Adds random gaussian noise multiplied by a weight to a random entry,
    /// simultaneously ensuring that the value is within the set bounds
    pub fn mutate_single_value(&mut self, w: f32, bounds: (f32, f32)) -> (usize, usize) {
        let (x, y) = self.random_entry();

        self.data[[x, y]] = random::gaussian(self.data[[x,y]], w, bounds);

        (x, y)
    }

    /// Crossover by dividing the matrix into 4 quadrants by a point,
    /// then selecting 2 quadrants from each.
    pub fn point_crossover(&self, other: &Self) -> Self {
        let mut new_data = self.data.clone();

        let (x, y) = self.random_entry();

        new_data.slice_mut(s![0..x, y..]).assign(&other.data.slice(s![0..x, y..]));
        new_data.slice_mut(s![x.., 0..y]).assign(&other.data.slice(s![x.., 0..y]));

        MatrixGene {
            data: new_data
        }
    }

    /// Crossover where for each row, we select the first k elements
    /// from the first genome and the len-k elements from the second genome
    pub fn point_row_crossover(&self, other: &Self) -> Self {
        let len = self.data.shape()[0];

        let mut new_data = Array::zeros((len, len));

        for i in 0..len {
            let k = random::random_range((0, len));

            new_data.slice_mut(s![i,..k])
                .assign(&self.data.slice(s![i,..k]));
            new_data.slice_mut(s![i,k..])
                .assign(&other.data.slice(s![i,k..]));
        }

        MatrixGene {
            data: new_data
        }
    }

    /// Crossover by randomly selecting rows from each matrix
    pub fn row_crossover(&self, other: &Self) -> Self {
        let len = self.data.shape()[0];

        let mut new_data = Array::zeros((len, len));

        for i in 0..len {
            if random::random_range((0.0, 1.0)) < 0.5 {
                new_data.slice_mut(s![i,..])
                    .assign(&self.data.slice(s![i,..]));
            } else {
                new_data.slice_mut(s![i,..])
                    .assign(&other.data.slice(s![i,..]));
            }
        }

        MatrixGene {
            data: new_data
        }
    }

    fn random_entry(&self) -> (usize, usize) {
        let len = self.data.shape()[0];

        let x = random::random_range((0, len));
        let y = random::random_range((0, len));

        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 10;

    #[test]
    fn matrix_gene_single_value_mutation() {
        let m1 = MatrixGene {data: Array::ones((N, N))};
        let mut m2 = m1.clone();

        for _ in 0..100 {
            m2.mutate_single_value(0.1, (0.0, 1.0));
        }

        assert!(m1.data != m2.data);
    }

    #[test]
    fn matrix_gene_mutate_within_bounds() {
        let mut m = MatrixGene {data: Array::ones((N, N))};

        for _ in 0..100 {
            let (x, y) = m.mutate_single_value(0.1, (0.0, 1.0));
            assert!(m.data[[x, y]] >= 0.0 && m.data[[x, y]] <= 1.0);
        }
    }

    #[test]
    fn matrix_gene_point_crossover_results_in_differing_rows() {
        let m1 = MatrixGene {data: Array::ones((N, N))};
        let m2 = MatrixGene {data: Array::zeros((N, N))};

        let m3 = m1.point_crossover(&m2);

        for i in 0..m3.data.shape()[0] {
            assert!(m3.data.slice(s![i,..]) != m2.data.slice(s![i,..])
                || m3.data.slice(s![i,..]) != m1.data.slice(s![i,..]));
        }
    }

    #[test]
    fn matrix_gene_row_crossover_results_in_identical_rows() {
        let m1 = MatrixGene {data: Array::ones((N, N))};
        let m2 = MatrixGene {data: Array::zeros((N, N))};

        let m3 = m1.row_crossover(&m2);

        for i in 0..m3.data.shape()[0] {
            assert!(m3.data.slice(s![i,..]) == m2.data.slice(s![i,..])
                || m3.data.slice(s![i,..]) == m1.data.slice(s![i,..]));
        }
    }
}
