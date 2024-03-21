use ndarray::{s, Array, Array1, Array2};

use crate::model::{NeuronModel, FiringState};


pub struct Izhikevich {
    a: Array1<f32>,
    b: Array1<f32>,
    c: Array1<f32>,
    d: Array1<f32>,
    v: Array1<f32>,
    u: Array1<f32>,
}

impl NeuronModel for Izhikevich {
    /**
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

    fn step(&mut self, input: Array1<f32>) -> FiringState {
        assert!(input.shape()[0] == self.v.shape()[0]);

        self.reset();

        // Numerically integrate the model
        *&mut self.v = &self.v + 0.5 * (0.04 * (&self.v * &self.v) + 5.0 * &self.v + 140.0 - &self.u + &input);
        *&mut self.v = &self.v + 0.5 * (0.04 * (&self.v * &self.v) + 5.0 * &self.v + 140.0 - &self.u + &input);

        *&mut self.u = &self.a * (&self.b * &self.v - &self.u);

        FiringState {
            state: self.v.mapv(|i| if i >= 30.0 {1.0} else {0.0})
        }
    }

    fn potentials(&self) -> Array1<f32> {
        self.v.to_owned()
    }
}


impl Izhikevich {
    pub fn new(n: usize, params: Array2<f32>) -> Izhikevich {
        assert!(params.shape() == [n, 4]);

        // Unpack parameter matrix for ease of use
        let a = params.slice(s![..,0]).to_owned();
        let b = params.slice(s![..,1]).to_owned();
        let c = params.slice(s![..,2]).to_owned();
        let d = params.slice(s![..,3]).to_owned();

        let potential: Array1<f32> = c.to_owned();
        let recovery: Array1<f32> = d.to_owned();

        Izhikevich {
            v: potential,
            u: recovery,
            a,
            b,
            c,
            d
        }
    }

    pub fn default(n: usize) -> Izhikevich {
        let izh_default: Vec<f32> = vec!(0.02, 0.2, -65.0, 2.0);
        let params: Array2<f32> = Array::from_shape_fn((n,4), |(_,j)| izh_default[j]);

        Self::new(n, params)
    }

    fn reset(&mut self) {
        /*
         * Reset neurons with v >= 30
         */

        // TODO: This can be parallelized w/ rayon
        for i in 0..self.v.shape()[0] {
            if self.v[i] >= 30.0 {
                *&mut self.v[i] = self.c[i];
                *&mut self.u[i] = self.u[i] + self.d[i];
            }
        }
    }
}

