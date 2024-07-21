pub mod er_model;

use crate::csa::DynamicsSet;
use crate::csa::mask::Mask;

use model::network::representation::{DefaultRepresentation, NetworkRepresentation, NeuronDescription};
use model::neuron::izhikevich::IzhikevichParameters;

use utils::config::Configurable;
use utils::environment::Environment;

use ndarray::{Array, Array2};


struct Dynamics {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    i: bool,
}

pub trait SRSNN: Configurable {
    fn new(c: Self::Config) -> Self;

    fn dynamics(&self) -> DynamicsSet;
    fn connectivity(&self) -> Mask;

    fn sample(&self, n: usize, env: &Environment) -> DefaultRepresentation {
        let mask = self.connectivity();
        let dynamics = self.dynamics();

        let network_cm = mask.matrix(n);
        let d = dynamics.vec(n);

        let mut neurons = Vec::new();

        for i in 0..n {
            let inhibitory = if d[i][4] == 1.0 { true } else { false };
            neurons.push(NeuronDescription::new(
                    i as u32,
                    IzhikevichParameters {
                        a: d[i][0],
                        b: d[i][1],
                        c: d[i][2],
                        d: d[i][3],
                    },
                    inhibitory,
            ));
        }

        let network_w = network_cm.mapv(|v| v as f32);

        let input_cm: Array2<u32> = Array::ones((n, env.inputs));
        let input_w: Array2<f32> = Array::ones((n, env.inputs));

        NetworkRepresentation::new(neurons.into(), network_cm, network_w, input_cm, input_w, env.clone())
    }

    //fn evolvable_parameters(6self) -> Vec<&mut dyn EvolvableParameter>;
}
