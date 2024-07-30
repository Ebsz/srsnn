use crate::models::rsnn::{RSNN, RSNNConfig};

use csa::{ConnectionSet, ValueSet, NeuronSet};

use utils::math;
use utils::config::{ConfigSection, Configurable};

use utils::parameters::{Parameter, ParameterSet};

use serde::Deserialize;
use ndarray::{array, Array, Array1, Array2};

use std::sync::Arc;


#[derive(Clone, Debug)]
pub struct TypedModel {
    //t_cpm: Array2<f32>,
    //dist: Vec<usize>
}

impl RSNN for TypedModel {
    fn params(config: &RSNNConfig<Self>) -> ParameterSet {
        let t_cpm = Parameter::Matrix(Array::zeros((config.model.k, config.model.k)));
        let t_w = Parameter::Matrix(Array::zeros((config.model.k, config.model.k)));
        let p = Parameter::Vector(Array::zeros(config.model.k));

        ParameterSet {
            set: vec![t_cpm, t_w, p],
        }
    }

    fn connectivity(params: &ParameterSet, config: &RSNNConfig<Self>) -> ConnectionSet {
        let (m1, m2, v) = Self::parse_params(params);

        let t_cpm = m1.mapv(|x| math::sigmoid(x));
        let t_w = m2.mapv(|x| math::sigmoid(x) * config.model.max_w);
        let p = math::softmax(v);

        let dist = math::distribute(config.n, p.as_slice().unwrap());

        let labels = csa::op::label(dist, config.model.k);

        let l = labels.clone();
        let w = ValueSet { f: Arc::new(
            move |i, j| t_w[[l(i) as usize, l(j) as usize]]
        )};

        ConnectionSet {
            m: csa::op::sbm(labels, ValueSet::from_value(t_cpm.clone())), // Do we have to clone?
            v: vec![ w ]
        }
    }

    fn dynamics(p: &ParameterSet, config: &RSNNConfig<Self>) -> NeuronSet {
        NeuronSet { f: Arc::new(
            move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        )}
    }
}

impl TypedModel {
    fn parse_params(p: &ParameterSet) -> (&Array2<f32>, &Array2<f32>, &Array1<f32>) {
        let a = match &p.set[0] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };
        let b = match &p.set[1] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };
        let c = match &p.set[2] {
            Parameter::Vector(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        (a, b, c)
    }
}

impl Configurable for TypedModel {
    type Config = TypedConfig;
}

#[derive(Clone, Debug, Deserialize)]
pub struct TypedConfig {
    pub k: usize,
    pub max_w: f32
}

impl ConfigSection for TypedConfig {
    fn name() -> String {
        "typed_model".to_string()
    }
}
