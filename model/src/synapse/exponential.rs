/// Single exponential synapse

use crate::spikes::Spikes;
use crate::synapse::Synapse;

use ndarray::{Array, Array1, Array2};



const TAU: f32 = 10.0;

pub struct ExponentialSynapse {
    w: Array2<f32>,
    neuron_type: Array1<f32>,

    s: Array1<f32>
}

impl Synapse for ExponentialSynapse {
    fn new(w: Array2<f32>, neuron_type: Array1<f32>) -> Self {
        ExponentialSynapse {
            s: Array::zeros(w.shape()[0]),

            w,
            neuron_type,
        }
    }

    fn step(&mut self, input: &Spikes) -> Array1<f32> {
        let ns = &input.into() * &self.neuron_type;

        self.s = &self.s + ( -&self.s / TAU + self.w.dot(&ns));

        log::trace!("ExponentialSynapse: {}", self.s);
        self.s.clone()
    }

    fn shape(&self) -> (usize, usize) {
        (self.w.shape()[0], self.w.shape()[1])
    }

    fn reset(&mut self) {
        self.s = Array::zeros(self.w.shape()[0]);
    }
}
