use std::collections::HashMap;

use ndarray::Array1;

pub const P_TOLERANCE: f32 = 1.0001;


/// Return the index of the entry in the array with the max value
pub fn max_index<T: PartialOrd>(data: &Array1<T>) -> usize {
    data.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).expect("")).map(|(k, _)| k).unwrap()
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
pub fn distribute(n: usize, p: &Vec<f32>) -> Vec<usize> {
    assert!(p.iter().all(|p| *p >= 0.0 && *p <= 1.0));
    assert!(p.iter().sum::<f32>() <= P_TOLERANCE, "expected p.sum <= 1.0, was {}", p.iter().sum::<f32>());

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
}
