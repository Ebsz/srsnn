use ndarray::{s, Axis, Array, Array1, Array2};


pub type SpikeSeries = Array2<f32>;

pub fn to_firing_rate(data: SpikeSeries) -> Array2<f32> {
    const time_window: usize = 10; // number of timesteps we average over

    let max_t = data.nrows();

    let mut fr: Array2<f32> = Array::zeros(data.raw_dim());

    for t in 0..max_t {
        let t2 = std::cmp::min(t+time_window, max_t) as usize;

        let d: Array1<f32> = data.slice(s!(t..t2, ..)).sum_axis(Axis(0)).mapv(|x| x as f32 / max_t as f32) * 1000.0;

        fr.slice_mut(s!(t,..)).assign(&d);
    }

    fr
}
