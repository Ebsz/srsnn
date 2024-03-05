use crate::model::NeuronModel;
use crate::izhikevich::Izhikevich;
use crate::synapses::{Synapses, MatrixSynapse};

use crate::population::Population;

use ndarray::{Array2, Array};

pub struct IzhikevichPool<S: Synapses> {
    neurons: Izhikevich,
    synapses: S 
}

impl<S: Synapses> Population<Izhikevich, S> for IzhikevichPool<S> {
    fn model(&mut self) -> &mut dyn NeuronModel {
        &mut self.neurons
    }

    fn synapses(&mut self) -> &mut dyn Synapses {
        &mut self.synapses
    }
}

impl IzhikevichPool<MatrixSynapse> {
    pub fn matrix_pool(n: usize) -> IzhikevichPool<MatrixSynapse> {
        let izh_default: Vec<f32> = vec!(0.02, 0.2, -65.0, 2.0);
        let params: Array2<f32> = Array::from_shape_fn((n,4), |(_,j)| izh_default[j]);

        let izh = Izhikevich::new(n, params);
        let synapses = MatrixSynapse::new(n);

        IzhikevichPool {
            neurons: izh,
            synapses

        }
    }
}
