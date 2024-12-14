/// Double exponential synapse, modeling both current rise and fall.

use crate::spikes::Spikes;
use crate::synapse::Synapse;

use ndarray::{Array, Array1, Array2};


const TAU_D: f32 = 5.0;
const TAU_R: f32 = 8.0;

pub struct BiExponentialSynapse {
    w: Array2<f32>,
    neuron_type: Array1<f32>,

    s: Array1<f32>,
    h: Array1<f32>
}

impl Synapse for BiExponentialSynapse {
    fn new(w: Array2<f32>, neuron_type: Array1<f32>) -> Self {
        let n_in = w.shape()[0];

        BiExponentialSynapse {
            s: Array::zeros(n_in),
            h: Array::zeros(n_in),

            w,
            neuron_type,
        }
    }

    fn step(&mut self, input: &Spikes) -> Array1<f32> {
        let ns = &input.into() * &self.neuron_type;

        self.s = &self.s + ( -&self.s / TAU_D + &self.h);

        self.h = &self.h + ( -&self.h / TAU_R  + 1.0 / ( TAU_D + TAU_R) * self.w.dot(&ns));

        log::trace!("BiExponential: {}", self.s);
        self.s.clone()
    }

    fn shape(&self) -> (usize, usize) {
        (self.w.shape()[0], self.w.shape()[1])
    }

    fn reset(&mut self) {
        self.s = Array::zeros(self.w.shape()[0]);
        self.h = Array::zeros(self.w.shape()[0]);
    }
}
