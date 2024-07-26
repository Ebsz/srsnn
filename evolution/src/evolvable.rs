use utils::parameters::Parameter;


trait Evolvable {
    fn mutate(&mut self);

    fn crossover(&self, other: &Self) -> Self;
}


impl Evolvable for Parameter {

    fn mutate(&mut self) {

    }

    fn crossover(&self, other: &Self) -> Self {

        self.clone()
    }
}


//pub struct ParameterSet(Vec<Box<dyn Parameter>>);
//
//impl ParameterSet {
//    pub fn size(&self) -> usize {
//        self.0.iter().map(|p| p.len()).sum()
//    }
//}
//
//trait Parameter {
//    fn len(&self) -> usize;
//}
//
//pub struct Scalar<T> {
//    data: T
//}
//
//impl<T> Parameter for Scalar<T> {
//    fn len(&self) -> usize {
//        1
//    }
//}
//
//impl<T> Parameter for Vector<T> {
//    fn len(&self) -> usize {
//        self.data.len()
//    }
//}
//
//impl<T> Parameter for Matrix<T> {
//    fn len(&self) -> usize {
//        self.data.len()
//    }
//}
//
//pub struct Vector<T> {
//    data: Array1<T>
//}
//
///// Generic matrix representation for arbitrary kinds of data,
//#[derive(Clone, Debug)]
//pub struct Matrix<T> {
//    pub data: Array2<T>,
//}
//
//impl Matrix<f32> {
//    pub fn init_random(n: usize, range: (f32, f32)) -> Matrix<f32> {
//        Matrix {
//            data: random::random_matrix((n, n), Uniform::new(range.0, range.1))
//        }
//    }
//
//    /// Adds random gaussian noise multiplied by a weight to a random entry,
//    /// simultaneously ensuring that the value is within the set bounds
//    pub fn mutate_single_value(&mut self, w: f32, bounds: (f32, f32)) -> (usize, usize) {
//        let (x, y) = self.random_entry();
//
//        self.data[[x, y]] = random::gaussian(self.data[[x,y]], w, bounds);
//
//        (x, y)
//    }
//
//    /// Crossover by dividing the matrix into 4 quadrants by a point,
//    /// then selecting 2 quadrants from each.
//    pub fn point_crossover(&self, other: &Self) -> Self {
//        let mut new_data = self.data.clone();
//
//        let (x, y) = self.random_entry();
//
//        new_data.slice_mut(s![0..x, y..]).assign(&other.data.slice(s![0..x, y..]));
//        new_data.slice_mut(s![x.., 0..y]).assign(&other.data.slice(s![x.., 0..y]));
//
//        Matrix {
//            data: new_data
//        }
//    }
//
//    /// Crossover where for each row, we select the first k elements
//    /// from the first genome and the len-k elements from the second genome
//    pub fn point_row_crossover(&self, other: &Self) -> Self {
//        let len = self.data.shape()[0];
//
//        let mut new_data = Array::zeros((len, len));
//
//        for i in 0..len {
//            let k = random::random_range((0, len));
//
//            new_data.slice_mut(s![i,..k])
//                .assign(&self.data.slice(s![i,..k]));
//            new_data.slice_mut(s![i,k..])
//                .assign(&other.data.slice(s![i,k..]));
//        }
//
//        Matrix {
//            data: new_data
//        }
//    }
//
//    /// Crossover by randomly selecting rows from each matrix
//    pub fn row_crossover(&self, other: &Self) -> Self {
//        let len = self.data.shape()[0];
//
//        let mut new_data = Array::zeros((len, len));
//
//        for i in 0..len {
//            if random::random_range((0.0, 1.0)) < 0.5 {
//                new_data.slice_mut(s![i,..])
//                    .assign(&self.data.slice(s![i,..]));
//            } else {
//                new_data.slice_mut(s![i,..])
//                    .assign(&other.data.slice(s![i,..]));
//            }
//        }
//
//        Matrix {
//            data: new_data
//        }
//    }
//
//    fn random_entry(&self) -> (usize, usize) {
//        let len = self.data.shape()[0];
//
//        let x = random::random_range((0, len));
//        let y = random::random_range((0, len));
//
//        (x, y)
//    }
//}
//
//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    const N: usize = 10;
//
//    #[test]
//    fn matrix_gene_single_value_mutation() {
//        let m1 = Matrix {data: Array::ones((N, N))};
//        let mut m2 = m1.clone();
//
//        for _ in 0..100 {
//            m2.mutate_single_value(0.1, (0.0, 1.0));
//        }
//
//        assert!(m1.data != m2.data);
//    }
//
//    #[test]
//    fn matrix_gene_mutate_within_bounds() {
//        let mut m = Matrix {data: Array::ones((N, N))};
//
//        for _ in 0..100 {
//            let (x, y) = m.mutate_single_value(0.1, (0.0, 1.0));
//            assert!(m.data[[x, y]] >= 0.0 && m.data[[x, y]] <= 1.0);
//        }
//    }
//
//    #[test]
//    fn matrix_gene_point_crossover_results_in_differing_rows() {
//        let m1 = Matrix {data: Array::ones((N, N))};
//        let m2 = Matrix {data: Array::zeros((N, N))};
//
//        let m3 = m1.point_crossover(&m2);
//
//        for i in 0..m3.data.shape()[0] {
//            assert!(m3.data.slice(s![i,..]) != m2.data.slice(s![i,..])
//                || m3.data.slice(s![i,..]) != m1.data.slice(s![i,..]));
//        }
//    }
//
//    #[test]
//    fn matrix_gene_row_crossover_results_in_identical_rows() {
//        let m1 = Matrix {data: Array::ones((N, N))};
//        let m2 = Matrix {data: Array::zeros((N, N))};
//
//        let m3 = m1.row_crossover(&m2);
//
//        for i in 0..m3.data.shape()[0] {
//            assert!(m3.data.slice(s![i,..]) == m2.data.slice(s![i,..])
//                || m3.data.slice(s![i,..]) == m1.data.slice(s![i,..]));
//        }
//    }
//}
