use ndarray::{Array, Array1};

use crate::neuron::NeuronModel;
use crate::spikes::Spikes;


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

    params: IzhikevichParameters
}

pub struct IzhikevichParameters {
    a: Array1<f32>,
    b: Array1<f32>,
    c: Array1<f32>,
    d: Array1<f32>,
}

impl IzhikevichParameters {
    fn n_default(n: usize) -> IzhikevichParameters {
        const DEFAULTS: (f32, f32, f32, f32) = (0.02, 0.2, -65.0, 2.0);

        IzhikevichParameters {
            a: Array::ones(n) * DEFAULTS.0,
            b: Array::ones(n) * DEFAULTS.1,
            c: Array::ones(n) * DEFAULTS.2,
            d: Array::ones(n) * DEFAULTS.3,
        }
    }
}


impl NeuronModel for Izhikevich {
    type Parameters = IzhikevichParameters;

    fn step(&mut self, input: Array1<f32>) -> Spikes {
        assert!(input.shape()[0] == self.v.shape()[0]);

        self.reset_spiking();

        const INTEGRATION_STEPS: usize = 2;

        // Numerically integrate the model
        for _ in 0..INTEGRATION_STEPS {
            *&mut self.v = &self.v + 1.0/(INTEGRATION_STEPS as f32) * (0.04 * (&self.v * &self.v) + 5.0 * &self.v + 140.0 - &self.u + &input);
        }

        *&mut self.u = &self.params.a * (&self.params.b * &self.v - &self.u);

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
        self.v = self.params.c.to_owned();
        self.u = (&self.params.b * &self.v).to_owned();
    }

    fn potentials(&self) -> Array1<f32> {
        self.v.to_owned()
    }
}


impl Izhikevich {
    const THRESHOLD: f32 = 30.0;

    pub fn new(n: usize, params: IzhikevichParameters) -> Izhikevich {
        //assert!(params.shape() == [n, 4]);

        assert!(params.a.len() == n);
        assert!(params.b.len() == n);
        assert!(params.c.len() == n);
        assert!(params.d.len() == n);


        //// Unpack parameter matrix for ease of use
        //let a = params.slice(s![..,0]).to_owned();
        //let b = params.slice(s![..,1]).to_owned();
        //let c = params.slice(s![..,2]).to_owned();
        //let d = params.slice(s![..,3]).to_owned();

        let potential: Array1<f32> = params.c.to_owned();
        let recovery: Array1<f32> = (&params.b * &potential).to_owned();

        Izhikevich {
            v: potential,
            u: recovery,
            params
        }
    }


    fn reset_spiking(&mut self) {
        /*
         * Reset neurons with v >= 30
         */

        // TODO: This can be parallelized w/ rayon
        for i in 0..self.v.shape()[0] {
            if self.v[i] >= Self::THRESHOLD {
                *&mut self.v[i] = self.params.c[i];
                *&mut self.u[i] = self.u[i] + self.params.d[i];
            }
        }
    }

    pub fn default(n: usize) -> Izhikevich {
        Self::new(n, IzhikevichParameters::n_default(n))
    }
}
