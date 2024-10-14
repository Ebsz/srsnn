use ndarray::{s, Array, Array1, Array2, Axis};


pub fn firing_rate(spike_record: Array2<u32>, window: usize) -> Array2<f32> {
    let data = spike_record.map(|x| *x as f32);

    let max_t = data.nrows();

    let mut fr: Array2<f32> = Array::zeros(data.raw_dim());

    for t in 0..max_t as i32 {
        let t1 = std::cmp::max(t - window as i32, 0);
        //println!("{}, {}", t1, t- );

        let d: Array1<f32> = data.slice(s!(t1..t, ..))
            .sum_axis(Axis(0)).mapv(|x| x as f32);

        fr.slice_mut(s!(t,..)).assign(&d);

    }

    fr /= window as f32;

    fr
}
