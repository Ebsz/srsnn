pub fn clamp<T: PartialOrd>(val: T, min: T, max: T) -> T {
    if val < min {
        return min;
    }

    if val > max {
        return max;
    }

    val
}

#[test]
fn test_clamp_value() {
    assert!(clamp(0, 0, 1) == 0);
    assert!(clamp(-1, 0, 1) == 0);

    assert!(clamp(-10.0, 0.0, 1.0) == 0.0);

    assert!(clamp(4.6, 0.0, 1.0) == 1.0);
    assert!(clamp(4.6, 0.0, 10.0) == 4.6);
}
