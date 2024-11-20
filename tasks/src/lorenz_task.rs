use ndarray::{s, array, Array1, Array2};


/// Parameters are given as (rho, sigma, beta)
pub const FIXED_POINTS: (f64, f64, f64) = (60.0, 20.0, 8.0);
pub const CHAOS: (f64, f64, f64) = (36.0, 8.5, 3.5);
pub const LIMIT_CYCLES: (f64, f64, f64) = (35.0, 21.0, 1.0);
pub const DEFAULT: (f64, f64, f64) = (28.0, 10.0, 8.0 / 3.0);

const DT: f64 = 0.01;

pub fn simulate_default(steps: usize) -> Array2<f64> {
    simulate(steps, DEFAULT, (1.0, 1.0, 1.2))
}

pub fn simulate(steps: usize, params: (f64, f64, f64), init_values: (f64, f64, f64)) -> Array2<f64> {
    let mut p = Array2::zeros((steps + 1, 3));

    let (r, s, b) = params;
    let (x0, y0, z0) = init_values;

    p[[0,0]] = x0;
    p[[0,1]] = y0;
    p[[0,2]] = z0;

    for i in 0..steps {
        let x = p[[i, 0]];
        let y = p[[i, 1]];
        let z = p[[i, 2]];

        let ii = p.slice(s![i,..]);

        let r = &ii + lorenz(x, y, z, s, r, b) * DT;

        let mut v = p.slice_mut(s![i+1, ..]);
        v.assign(&r);
    }

    p
}

fn lorenz(x: f64, y: f64, z: f64, s: f64, r: f64, b: f64) -> Array1<f64> {
    let x_dot = s * (y - x);
    let y_dot = x * (r - z) - y;
    let z_dot = x * y - b * z;

    array![x_dot, y_dot, z_dot]
}
