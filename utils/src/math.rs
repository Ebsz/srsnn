use std::collections::HashMap;

use ndarray::Array1;


pub const P_TOLERANCE: f32 = 1.0001;

pub fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let mut s = x.mapv(f32::exp);

    s /= s.sum();

    s
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

// min/max for floats
pub fn maxf(data: &Vec<f32>) -> f32 {
    data.iter().fold(0.0f32, |acc, &x| if x > acc {x} else {acc})
}
pub fn minf(data: &Vec<f32>) -> f32 {
    data.iter().fold(0.0f32, |acc, &x| if x < acc {x} else {acc})
}

/// Return the index of the entry in the array with the max value
pub fn max_index<T: PartialOrd, I: IntoIterator<Item = T>>(data: I) -> usize {
    data.into_iter().enumerate().max_by(|a,b| a.1.partial_cmp(&b.1).expect("")).map(|(k, _)| k).unwrap()
}

pub fn clamp<T: PartialOrd>(val: T, min: T, max: T) -> T {
    if val < min {
        return min;
    }

    if val > max {
        return max;
    }

    val
}

/// Distribute n items into k = |p| buckets with probability distribution p
/// NOTE: This is not deterministic.
pub fn distribute(n: usize, p: &[f32]) -> Vec<usize> {
    assert!(p.iter().all(|p| *p >= 0.0 && *p <= 1.0), "distribute() error in p: {:#?}", p);
    assert!(p.iter().sum::<f32>() <= P_TOLERANCE, "distribute() expected p.sum <= 1.0, was {}", p.iter().sum::<f32>());

    let mut buckets: Vec<usize> = p.iter().map(|p| (p * n as f32).floor() as usize).collect();
    let sum_buckets = buckets.iter().sum::<usize>();

    assert!(sum_buckets <= n);

    // Assign the remaining items by minimizing the distance between desired and actual fractions
    let remainder: usize = n - sum_buckets;
    for _ in 0..remainder {
        let dist: HashMap<usize, f32> = buckets.iter().enumerate().zip(p).map(|((i, b), p)| (i, p - (*b as f32 / n as f32))).collect();
        let max_idx: usize = *dist.iter().max_by(|a, b| a.1.partial_cmp(b.1).expect("")).map(|(k, _)| k).unwrap();

        buckets[max_idx] += 1;
    }

    assert!(buckets.iter().sum::<usize>() == n);

    buckets
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_clamp_value() {
        assert!(clamp(0, 0, 1) == 0);
        assert!(clamp(-1, 0, 1) == 0);

        assert!(clamp(-10.0, 0.0, 1.0) == 0.0);

        assert!(clamp(4.6, 0.0, 1.0) == 1.0);
        assert!(clamp(4.6, 0.0, 10.0) == 4.6);
    }

    #[test]
    fn test_distribute() {
        let k = vec![0.21, 0.29, 0.14, 0.36];

        for n in [4, 10, 128, 199, 256, 421] {
            let dist = distribute(n, &k);
            assert!(dist.iter().sum::<usize>() == n);
        }
    }

    #[test]
    fn test_max_index() {
        let v = vec![0,1,2,3,4,5,6,7,8,9];
        let a = array![0,1,2,3,4,5,6,7,8,9];

        assert!(max_index(v) == 9);
        assert!(max_index(a) == 9);
    }
}
