//! Geometric typed model

use crate::models::rsnn::{RSNN, RSNNConfig};

use csa::op::{LabelFn, CoordinateFn, Metric};
use csa::{ConnectionSet, ValueSet, NeuronSet, NetworkSet};
use csa::mask::Mask;

use model::neuron::izhikevich::IzhikevichParameters;

use utils::{math, random};
use utils::config::{ConfigSection, Configurable};
use utils::parameters::{Parameter, ParameterSet};
use utils::environment::Environment;

use serde::Deserialize;

use ndarray::{s, Array, Array1, Array2};

use std::sync::Arc;


// with N(0,1), 0.7 gives a starting inhibitory prob of ~0.2
const INHIBITORY_THRESHOLD: f32 = 0.70;

#[derive(Clone, Debug)]
pub struct GeometricTypedModel;

impl RSNN for GeometricTypedModel {
    fn params(config: &RSNNConfig<Self>, env: &Environment) -> ParameterSet {
        // Type connection probability matrix
        let t_cpm = Parameter::Matrix(Array::zeros((config.model.k, config.model.k)));

        // Type weights
        let t_w = Parameter::Matrix(Array::zeros((config.model.k, config.model.k)));

        // Probability distribution over the k types
        let p = Parameter::Vector(Array::zeros(config.model.k));

        // Dynamical parameters
        let d = Parameter::Matrix(Array::zeros((config.model.k, Self::N_DYNAMICAL_PARAMETERS)));

        // Input->type connection probabilities
        let input_t_cp = Parameter::Vector(Array::zeros(config.model.k));

        // Input->type weights
        let input_t_w = Parameter::Vector(Array::zeros(config.model.k));

        // Type->output connection probabilities
        let output_t_cp = Parameter::Vector(Array::zeros(config.model.k));

        ParameterSet {
            set: vec![t_cpm, t_w, p, d, input_t_cp, input_t_w, output_t_cp],
        }
    }

    fn get(params: &ParameterSet, config: &RSNNConfig<Self>) -> (NetworkSet, ConnectionSet, Mask) {
        let (m1, m2, v1, m3, v2, v3, v4) = Self::parse_params(params, config);

        let t_cpm = m1.mapv(|x| math::ml::sigmoid(x));

        let t_w = m2.mapv(|x| math::ml::sigmoid(x) * config.model.max_w);
        let p = math::ml::softmax(v1);

        assert!(!p.iter().all(|x| x.is_nan()), "p contained NaN - p: {p}, v1: {v1}");

        let p_test = p.as_slice().unwrap();
        assert!(!p_test.iter().all(|x| x.is_nan()), "p contained NaN - p: {p}, v1: {v1}");

        let dist = math::distribute(config.n, p.as_slice().unwrap());
        let labels = csa::op::label(dist, config.model.k);

        let l = labels.clone();
        let w = ValueSet { f: Arc::new(
            move |i, j| t_w[[l(i) as usize, l(j) as usize]]
        )};

        let dynamics = Self::get_dynamics(m3, labels.clone(), config);

        let sbm_mask = csa::op::sbm(labels.clone(), ValueSet::from_value(t_cpm.clone()));

        // geometric setup
        let coords: CoordinateFn = csa::op::random_coordinates(0.0, config.model.max_coordinate, config.n);
        let d: Metric = csa::op::distance_metric(coords.clone(), coords.clone());

        let disc = csa::op::disc(config.model.distance_threshold, d);

        let mask = sbm_mask & disc;

        let ns = NetworkSet {
            m: mask,
            v: vec![w],
            d: vec![dynamics]
        };

        let input_cs = Self::get_input_cs(v2, v3, labels.clone(), coords.clone(), config);
        let output_mask = Self::get_output_mask(v4, labels, coords.clone(), config);

        (ns, input_cs, output_mask)
    }
}

impl GeometricTypedModel {
    fn parse_params<'a>(p: &'a ParameterSet, config: &'a RSNNConfig<Self>)
        -> (&'a Array2<f32>, &'a Array2<f32>, &'a Array1<f32>, &'a Array2<f32>,
            &'a Array1<f32>, &'a Array1<f32>, &'a Array1<f32>) {
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

        let e = match &p.set[4] {
            Parameter::Vector(x) => {x},
            _ => { panic!("invalid parameter set") }
        };
        let f = match &p.set[5] {
            Parameter::Vector(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        let g = match &p.set[6] {
            Parameter::Vector(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        assert!(a.shape() == [config.model.k, config.model.k]);
        assert!(b.shape() == [config.model.k, config.model.k]);
        assert!(c.shape() == [config.model.k]);
        assert!(d.shape() == [config.model.k, Self::N_DYNAMICAL_PARAMETERS]);
        assert!(e.shape() == [config.model.k]);
        assert!(f.shape() == [config.model.k]);
        assert!(g.shape() == [config.model.k]);

        (a, b, c, d, e, f, g)
    }

    fn get_dynamics(m: &Array2<f32>, l: LabelFn, config: &RSNNConfig<Self>) -> NeuronSet {
        let mut dm = m.mapv(|x| math::ml::sigmoid(x));

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

    fn get_input_cs(v2: &Array1<f32>, v3: &Array1<f32>, l: LabelFn, g: CoordinateFn, config: &RSNNConfig<Self>)
        -> ConnectionSet {
        let input_t_cp = v2.mapv(|x| math::ml::sigmoid(x));
        let input_t_w = v3.mapv(|x| math::ml::sigmoid(x) * config.model.max_w);

        let cp = ValueSet { f: Arc::new(
            move |i, _| input_t_cp[i as usize]
            )};

        let l2 = l.clone();
        let m = Mask {
            f: Arc::new(
                   move |i, j| random::random_range((0.0, 1.0)) < (cp.f)(l2(i), j)
               )
        };

        let input_g: CoordinateFn = Arc::new(move |i| (0.0, 0.0));
        let d: Metric = csa::op::distance_metric(g, input_g); // NOTE: Ensure this is correct

        let input_mask = m & csa::op::disc(config.model.distance_threshold, d);

        let w = ValueSet { f: Arc::new(
            move |i, _| input_t_w[[l(i) as usize]]
        )};

        ConnectionSet {
            m: input_mask,
            v: vec![w]
        }
    }

    fn get_output_mask(v4: &Array1<f32>, l: LabelFn, g: CoordinateFn, config: &RSNNConfig<Self>) -> Mask {
        let output_t_cp = v4.mapv(|x| math::ml::sigmoid(x));

        let max = config.model.max_coordinate;
        let output_g: CoordinateFn = Arc::new(move |i| (max * 0.7, max * 0.7));

        let d: Metric = csa::op::distance_metric(output_g, g); // NOTE: Ensure this is correct

        let cp = ValueSet { f: Arc::new( move |_, j| output_t_cp[j as usize])};

        let m = Mask { f: Arc::new( move |i, j| random::random_range((0.0, 1.0)) < (cp.f)(i,l(j))) };

        m & csa::op::disc(config.model.distance_threshold, d)
    }
}

impl Configurable for GeometricTypedModel {
    type Config = GeometricTypedConfig;
}

#[derive(Clone, Debug, Deserialize)]
pub struct GeometricTypedConfig {
    pub k: usize,
    pub max_w: f32,
    pub distance_threshold: f32,
    pub max_coordinate: f32
}

impl ConfigSection for GeometricTypedConfig {
    fn name() -> String {
        "typed_model".to_string()
    }
}
