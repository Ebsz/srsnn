//! Integrate-and-fire neuron
//!

use crate::spikes::Spikes;
use crate::neuron::NeuronModel;

use ndarray::{Array, Array1};

use serde::{Serialize, Deserialize};


const DEFAULT_R: f32 = 5e7f32;
const DEFAULT_C: f32 = 1e-10f32;

const TIME_STEP: f32 = 0.001;
const FIRING_THRESHOLD: f32 = 1.0;

pub struct Lapicque {
    n: usize,

    pub v: Array1<f32>,

    r: Array1<f32>,
    c: Array1<f32>,
}

impl NeuronModel for Lapicque {
    type Parameters = LapicqueParameters;

    fn new(n: usize, params: Vec<LapicqueParameters>) -> Self {
        Lapicque {
            n,
            v: Array::zeros(n),

            r: params.iter().map(|p| p.r).collect(),
            c: params.iter().map(|p| p.c).collect(),
        }
    }

    fn step(&mut self, input: Array1<f32>) -> Spikes {
        assert!(input.shape()[0] == self.v.shape()[0]);

        // Reset neurons with v >= firing threshold
        self.v = self.v.mapv(|p| if p >= FIRING_THRESHOLD { 0.0 } else { p });


        let tau = &self.r * &self.c;

        const INTEGRATION_STEPS: usize = 2;
        for _ in 0..INTEGRATION_STEPS {
            self.v = &self.v + (1.0 / INTEGRATION_STEPS as f32) * (TIME_STEP / &tau) * (-&self.v + &input * &self.r);
        }

        Spikes {
            data: self.v.mapv(|i| if i >= FIRING_THRESHOLD { true } else { false })
        }
    }

    fn reset(&mut self) {
        self.v = Array::zeros(self.n);
    }

    fn potentials(&self) -> Array1<f32> {
        self.v.clone()
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct LapicqueParameters {
    pub r: f32,
    pub c: f32,
}

impl Default for LapicqueParameters {
    fn default() -> Self {
        LapicqueParameters {
            r: DEFAULT_R,
            c: DEFAULT_C,
        }
    }
}
