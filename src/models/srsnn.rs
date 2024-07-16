pub mod er_model;

use crate::csa;
use crate::csa::DynamicsSet;
use crate::csa::mask::Mask;

use crate::models::stochastic::StochasticModel;

use model::network::representation::{DefaultRepresentation, NetworkRepresentation};

use utils::config::Configurable;

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

    //fn evolvable_parameters(6self) -> Vec<&mut dyn EvolvableParameter>;
}

pub trait SModel {
    fn sample(&self, n: usize) -> DefaultRepresentation;
}

//impl<S: SRSNN> SModel for S {
//    fn sample(&self, n: usize) -> DefaultRepresentation {
//        let mask = model.connectivity();
//        let dynamics = model.dynamics();
//
//        let m = mask.matrix(n);
//        let d = dynamics.vec(n);
//
//        //let neurons: Array1<NeuronDescription<Izhikevich>>;
//
//        NetworkRepresentation::new()
//    }
//}
