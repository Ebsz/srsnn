use criterion::{black_box, criterion_group, criterion_main, Criterion};

use luna::pools::Pool;
use model::network::Network;

use ndarray::{Array, Array2};


const T: usize = 1000;
const P: f32 = 0.1;
const INHIBITORY_FRACTION: f32 = 0.2;

const LARGE_POOL_SIZE: usize = 100;
const SMALL_POOL_SIZE: usize = 20;

/// Benchmark a pool of size N
fn run_pool(n: usize) {
    let input: Array2<f32> = Array::ones((T, n)) * 17.3;

    let mut network = Pool::new(n, P, INHIBITORY_FRACTION);
    let _ = network.run(T, &input);
}

//fn fibonacci(n: u64) -> u64 {
//    match n {
//        0 => 1,
//        1 => 1,
//        n => fibonacci(n-1) + fibonacci(n-2),
//    }
//}

//fn criterion_benchmark(c: &mut Criterion) {
//    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
//}

fn pool_benchmark(c: &mut Criterion) {
    c.bench_function("large pool", |b| b.iter(|| run_pool(black_box(LARGE_POOL_SIZE))));
    c.bench_function("small pool", |b| b.iter(|| run_pool(black_box(SMALL_POOL_SIZE))));
}

criterion_group!(benches, pool_benchmark);
criterion_main!(benches);
