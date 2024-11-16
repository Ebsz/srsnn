use crate::models::generator::{Generator, NetworkModel};
use csa::{ConnectionSet, ValueSet, NeuronSet};

use ndarray::array;

use std::sync::Arc;

pub struct ERModel;

impl Generator for ERModel {
    type Parameters = f32;

    fn g(p: f32) -> NetworkModel {
        let a = csa::mask::random(p);
        let w = ValueSet { f: Arc::new( move |_i, _j| 1.0) };

        let c = ConnectionSet { m: a, v: vec![w]};

        // Default dynamics
        let d = NeuronSet { f: Arc::new( move |i| array![0.02, 0.2, -65.0, 8.0, 0.0] ) };

        let a_i = csa::mask::random(p);
        let w_i = ValueSet { f: Arc::new( move |_i, _j| 1.0) };

        let i = ConnectionSet { m: a_i, v: vec![w_i]};

        NetworkModel (c, d, i)
    }
}

fn random_dynamics() {

}
