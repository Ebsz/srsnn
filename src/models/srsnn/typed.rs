use crate::models::rsnn::{RSNN, RSNNConfig};

use csa::op::LabelFn;
use csa::{ConnectionSet, ValueSet, NeuronSet, NeuralSet};

use model::neuron::izhikevich::IzhikevichParameters;

use utils::math;
use utils::config::{ConfigSection, Configurable};
use utils::parameters::{Parameter, ParameterSet};

use serde::Deserialize;

use ndarray::{s, array, Array, Array1, Array2};

use std::sync::Arc;


// with N(0,1), 0.7 gives a starting inhibitory prob of ~0.2
const INHIBITORY_THRESHOLD: f32 = 0.70;

#[derive(Clone, Debug)]
pub struct TypedModel;

impl RSNN for TypedModel {
    fn params(config: &RSNNConfig<Self>) -> ParameterSet {
        // Type connection probability matrix
        let t_cpm = Parameter::Matrix(Array::zeros((config.model.k, config.model.k)));

        // Type weights
        let t_w = Parameter::Matrix(Array::zeros((config.model.k, config.model.k)));

        // Probability distribution over the k types
        let p = Parameter::Vector(Array::zeros(config.model.k));

        // Dynamical parameters
        let d = Parameter::Matrix(Array::zeros((config.model.k, Self::N_DYNAMICAL_PARAMETERS)));

        ParameterSet {
            set: vec![t_cpm, t_w, p, d],
        }
    }

    fn get(params: &ParameterSet, config: &RSNNConfig<Self>) -> NeuralSet {
        let (m1, m2, v, m3) = Self::parse_params(params, config);

        let t_cpm = m1.mapv(|x| math::sigmoid(x-5.0));

        //log::info!("{:#?}", t_cpm.iter().sum::<f32>() / t_cpm.len() as f32);
        let t_w = m2.mapv(|x| math::sigmoid(x) * config.model.max_w);
        let p = math::softmax(v);

        let dist = math::distribute(config.n, p.as_slice().unwrap());
        let labels = csa::op::label(dist, config.model.k);

        let l = labels.clone();
        let w = ValueSet { f: Arc::new(
            move |i, j| t_w[[l(i) as usize, l(j) as usize]]
        )};

        // Create dynamics
        let d = Self::get_dynamics(m3, labels.clone(), config);

        let mask = csa::op::sbm(labels, ValueSet::from_value(t_cpm.clone()));

        //let d = NeuronSet { f: Arc::new(
        //    move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        //)};

        NeuralSet {
            m: mask,
            v: vec![w],
            d: vec![d]
        }
    }

    //fn connectivity(params: &ParameterSet, config: &RSNNConfig<Self>) -> ConnectionSet {
    //    let (m1, m2, v, m3) = Self::parse_params(params, config);

    //    let t_cpm = m1.mapv(|x| math::sigmoid(x));
    //    let t_w = m2.mapv(|x| math::sigmoid(x) * config.model.max_w);
    //    let p = math::softmax(v);

    //    let dist = math::distribute(config.n, p.as_slice().unwrap());

    //    let labels = csa::op::label(dist, config.model.k);

    //    let l = labels.clone();
    //    let w = ValueSet { f: Arc::new(
    //        move |i, j| t_w[[l(i) as usize, l(j) as usize]]
    //    )};

    //    ConnectionSet {
    //        m: csa::op::sbm(labels, ValueSet::from_value(t_cpm.clone())),
    //        v: vec![ w ]
    //    }
    //}
}

impl TypedModel {
    fn parse_params<'a>(p: &'a ParameterSet, config: &'a RSNNConfig<Self>)
        -> (&'a Array2<f32>, &'a Array2<f32>, &'a Array1<f32>, &'a Array2<f32>) {
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

        let d = match &p.set[3] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        assert!(a.shape() == [config.model.k, config.model.k]);
        assert!(b.shape() == [config.model.k, config.model.k]);
        assert!(c.shape() == [config.model.k]);
        assert!(d.shape() == [config.model.k, Self::N_DYNAMICAL_PARAMETERS]);

        (a, b, c, d)
    }

    fn get_dynamics(m: &Array2<f32>, l: LabelFn, config: &RSNNConfig<Self>) -> NeuronSet {
        let mut dm = m.mapv(|x| math::sigmoid(x));

        let r = IzhikevichParameters::RANGES;

        for mut vals in dm.rows_mut() {
            vals[0] = vals[0] * (r[0].1 - r[0].0) + r[0].0;
            vals[1] = vals[1] * (r[1].1 - r[1].0) + r[1].0;
            vals[2] = vals[2] * (r[2].1 - r[2].0) + r[2].0;
            vals[3] = vals[3] * (r[3].1 - r[3].0) + r[3].0;

            vals[4] = if vals[4] > INHIBITORY_THRESHOLD {1.0} else {0.0};
        }

        NeuronSet { f: Arc::new(
            move |i| dm.slice(s![l(i) as usize, ..]).to_owned()
        )}
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

#[cfg(test)]
mod tests {

    #[test]
    fn can_parse_params() {
        // TODO: Implement
        //
        // test normal params can be parsed ok
    }

    #[test]
    fn error_on_wrong_params() {
        // TODO: Implement
        //
        // test that calling parse_params() with wrong params panics

    }
}
