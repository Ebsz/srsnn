use crate::neuron::NeuronModel;
use crate::spikes::Spikes;

use ndarray::Array1;

use serde::{Serialize, Deserialize};


/*
 * The Izhikevich model captures the dynamics of neurons in a computationally feasible way,
 * and is described by the equations
 *
 *      dv/dt = 0.04 * v^2 + 5v + 140 -u + I
 *      du/dt = a(bv - u)
 *
 * and parameters a, b, c, d
 *
 * A neuron fires if v >= 30. Upon firing, the neuron is reset according to
 *
 *      v = c
 *      u = u + d
 */
pub struct Izhikevich {
    v: Array1<f32>,
    u: Array1<f32>,

    a: Array1<f32>,
    b: Array1<f32>,
    c: Array1<f32>,
    d: Array1<f32>,
}


impl NeuronModel for Izhikevich {
    type Parameters = IzhikevichParameters;

    fn new(n: usize, params: Vec<IzhikevichParameters>) -> Izhikevich {
        assert!(params.len() == n, "expected {n} params, got {:?}", params.len());

        let a: Array1<f32> = params.iter().map(|p| p.a).collect();
        let b: Array1<f32> = params.iter().map(|p| p.b).collect();
        let c: Array1<f32> = params.iter().map(|p| p.c).collect();
        let d: Array1<f32> = params.iter().map(|p| p.d).collect();

        let potential: Array1<f32> = c.to_owned();
        let recovery: Array1<f32> = (&b * &potential).to_owned();

        Izhikevich {
            v: potential,
            u: recovery,

            a,
            b,
            c,
            d
        }
    }

    fn step(&mut self, input: Array1<f32>) -> Spikes {
        assert!(input.shape()[0] == self.v.shape()[0]);

        self.reset_spiking();

        *&mut self.v = &self.v + 0.5 * (0.04 * (&self.v * &self.v) + 5.0 * &self.v + 140.0 - &self.u + &input);
        *&mut self.v = &self.v + 0.5 * (0.04 * (&self.v * &self.v) + 5.0 * &self.v + 140.0 - &self.u + &input);

        *&mut self.u = &self.u + &self.a * (&self.b * &self.v - &self.u);

        // Ensure potentials do not exceed the threshold value.
        // This has no effect on the model, but is necessary when using
        // the potentials in other contexts
        *&mut self.v = self.v.iter().map(|p| if *p > Self::THRESHOLD {Self::THRESHOLD} else {*p} ).collect();

        Spikes {
            data: self.v.mapv(|i| if i >= Self::THRESHOLD { true } else { false })
        }
    }

    /// Reset all neurons to their initial state
    fn reset(&mut self) {
        self.v = self.c.to_owned();
        self.u = (&self.b + &self.v).to_owned();
    }

    fn potentials(&self) -> Array1<f32> {
        self.v.to_owned()
    }
}


impl Izhikevich {
    const THRESHOLD: f32 = 30.0;

    fn reset_spiking(&mut self) {
        /*
         * Reset neurons with v >= 30
         */

        // TODO: This can be parallelized w/ rayon
        for i in 0..self.v.shape()[0] {
            if self.v[i] >= Self::THRESHOLD {
                *&mut self.v[i] = self.c[i];
                *&mut self.u[i] = self.u[i] + self.d[i];
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct IzhikevichParameters {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

impl IzhikevichParameters {
    pub const RS: Param = (0.02, 0.2, -65.0, 8.0);
    pub const FS: Param = (0.1, 0.2, -65.0, 2.0);
    pub const CH: Param = (0.02, 0.2, -50.0, 2.0);

    pub const REGULAR_SPIKING: Param = Self::RS;
    pub const FAST_SPIKING: Param = Self::FS;
    pub const CHATTERING: Param = Self::CH;

    // Default parameters correspond to regular spiking (RS) neurons
    pub const DEFAULT: Param = Self::RS;

    pub const RANGES: [(f32, f32); 4] =
        [(0.02, 0.1), (0.2, 0.25), (-65.0, -50.0), (2.0, 8.0)];
}

impl Default for IzhikevichParameters {
    fn default() -> Self {
        IzhikevichParameters {
            a: Self::DEFAULT.0,
            b: Self::DEFAULT.1,
            c: Self::DEFAULT.2,
            d: Self::DEFAULT.3,
        }
    }
}

type Param = (f32, f32, f32, f32);

impl From<(f32, f32, f32, f32)> for IzhikevichParameters {
    fn from(x: (f32, f32, f32, f32)) -> IzhikevichParameters {
        IzhikevichParameters { a: x.0, b: x.1, c: x.2, d: x.3 }
    }
}
