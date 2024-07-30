use crate::ValueSet;
use crate::mask::Mask;

use utils::random;

use std::sync::Arc;


type LabelFn = Arc<dyn Fn(u32) -> u32>;

/// The operator version of \rho, operating on the value set of connection probabilities
pub fn p(v: ValueSet) -> Mask {
    let f = v.f;
    Mask {
        f: Arc::new(
                move |i, j| random::random_range((0.0, 1.0)) < f(i, j)
            )
    }
}

/// Expands each entry in c to a nxn block in a new matrix
pub fn block(n: u32, m: Mask) -> Mask {
    let f1 = m.f.clone();

    Mask {
        f: Arc::new(
               move |i, j| f1(i/n, j/n)
           )
    }
}

pub fn label(dist: Vec<usize>, k: usize) -> LabelFn {
    let mut dist_map = vec![];
    for l in 0..k {
        dist_map.append(&mut vec![l; dist[l]]);
    }

    Arc::new(move |i| dist_map[i as usize] as u32)
}

pub fn sbm(l: LabelFn, cpm: ValueSet) -> Mask {
    let f = cpm.f;

    Mask {
        f: Arc::new(
               move |i, j| random::random_range((0.0, 1.0)) < f(l(i), l(j))
           )
    }
}

//fn sample(v: ValueSet<f32>) ->

//fn disc(r: f32, dist: impl Space) -> Mask {
//    Mask {
//        f: Arc::new(
//               move |i, j| dist(i, j) < r
//           )
//    }
//}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{array, s, Array, Array2};
    use utils::math;

    const N: usize = 10;

    #[test]
    fn mask_p_op() {
        let v0 = ValueSet::from_value(Array::zeros((N, N)));
        let v1 = ValueSet::from_value(Array::ones((N, N)));

        let m0 = (p(v0)).matrix(N);
        let m1 = (p(v1)).matrix(N);

        let t0: Array2<u32> = Array::zeros((N, N));
        let t1: Array2<u32> = Array::ones((N, N));

        assert!(m0 == t0);
        assert!(m1 == t1);
    }

    #[test]
    fn sbm_op() {
        let n = 10;
        let k = 2;
        let p = vec![0.5, 0.5];

        let dist = math::distribute(n, &p);
        println!("{:?}", dist);

        let labels = label(dist, k);

        let cpm = ValueSet::from_value(array![[0.0,1.0,],
                                              [1.0,0.0,]]);

        let m = sbm(labels, cpm);

        let mx = m.matrix(10);

        let a = mx.slice(s![..4,..4]); // top left
        let b = mx.slice(s![..4,(-4)..]); // top right
        let c = mx.slice(s![(-4)..,..4]); // bottom left
        let d = mx.slice(s![(-4)..,(-4)..]); // bottom right

        assert!(a.iter().all(|x| *x == 0));
        assert!(b.iter().all(|x| *x == 1));
        assert!(c.iter().all(|x| *x == 1));
        assert!(d.iter().all(|x| *x == 0));
    }
}
