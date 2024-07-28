use crate::csa;
use crate::csa::{ConnectionSet, ValueSet, DynamicsSet};
use crate::models::rsnn::{RSNN, RSNNConfig};

use utils::math;
use utils::config::{ConfigSection, Configurable};

use utils::parameters::{Parameter, ParameterSet};

use serde::Deserialize;
use ndarray::{array, Array, Array1, Array2};

use std::sync::Arc;


//struct TypedParameters {
//    n: Scalar<usize>,
//    k: Scalar<usize>,
//    t_cpm: Matrix<f32>,
//    t_dist: Vector<f32>,
//}

#[derive(Clone, Debug)]
pub struct TypedModel {
    //t_cpm: Array2<f32>,
    //dist: Vec<usize>
}

impl RSNN for TypedModel {
    fn params(config: &RSNNConfig<Self>) -> ParameterSet {
        let t_cpm = Parameter::Matrix(Array::zeros((config.model.k, config.model.k)));
        let p = Parameter::Vector(Array::zeros(config.model.k));

        ParameterSet {
            set: vec![t_cpm, p],
        }
    }

    fn connectivity(params: &ParameterSet, config: &RSNNConfig<Self>) -> ConnectionSet {
        let (t_cpm, p) = Self::parse_params(params);

        println!("{:?}", p);
        // Convert p from R^k to probability distribution over k

        let dist = math::distribute(config.n, p.as_slice().unwrap());

        let labels = csa::op::label(dist, config.model.k);

        ConnectionSet {
            m: csa::op::sbm(labels, ValueSet(t_cpm.clone())), // Do we have to clone?
            v: vec![Self::default_weights(config.n)]
        }
    }

    fn dynamics(p: &ParameterSet, config: &RSNNConfig<Self>) -> DynamicsSet {
        DynamicsSet { f: Arc::new(
            move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        )}
    }
}

impl TypedModel {
    fn parse_params(p: &ParameterSet) -> (&Array2<f32>, &Array1<f32>) {
        let a = match &p.set[0] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };
        let b = match &p.set[1] {
            Parameter::Vector(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        (a, b)
    }
}

impl Configurable for TypedModel {
    type Config = TypedConfig;
}

#[derive(Clone, Debug, Deserialize)]
pub struct TypedConfig {
    pub k: usize,
}

impl ConfigSection for TypedConfig {
    fn name() -> String {
        "typed_model".to_string()
    }
}
