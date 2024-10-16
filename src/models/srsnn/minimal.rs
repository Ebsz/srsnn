//! Minimal model

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
pub struct MinimalModel;

impl RSNN for MinimalModel {
    fn params(config: &RSNNConfig<Self>, _env: &Environment) -> ParameterSet {
        // Type connection probability matrix: [k + output]
        let t_cpm = Parameter::Matrix(Array::zeros((config.model.k + 1, config.model.k + 1)));

        // Probability distribution over the k types
        let p = Parameter::Vector(Array::zeros(config.model.k));

        // Type weights: [k + output, k + output]
        //let t_w = Parameter::Matrix(Array::zeros((config.model.k + 1, config.model.k + 1)));

        // Dynamical parameters
        //let d = Parameter::Matrix(Array::zeros((config.model.k, Self::N_DYNAMICAL_PARAMETERS)));

        // Separate vector of dynamical parameters for output type
        // without the inhibitory flag, because they are never inhibitory.
        //let d_out = Parameter::Vector(Array::zeros(Self::N_DYNAMICAL_PARAMETERS -1));

        // Input->type connection probabilities
        let input_t_cp = Parameter::Vector(Array::zeros(config.model.k));

        // Input->type weights
        // let input_t_w = Parameter::Vector(Array::zeros(config.model.k));

        ParameterSet {
            set: vec![t_cpm, p, input_t_cp],
        }
    }

    fn get(
        params: &ParameterSet,
        config: &RSNNConfig<Self>,
        env: &Environment)
        -> (NetworkSet, ConnectionSet)
    {
        let (m1, v1, v3) = Self::parse_params(params, config);
        let t_cpm = m1.mapv(|x| math::ml::sigmoid(x));

        let p = math::ml::softmax(v1);

        let p_test = p.as_slice().unwrap();
        assert!(!p.iter().all(|x| x.is_nan()), "p contained NaN - p: {p}, v1: {v1}");
        assert!(!p_test.iter().all(|x| x.is_nan()), "p contained NaN - p: {p}, v1: {v1}");

        //let t_w = m2.mapv(|x| math::ml::sigmoid(x) * config.model.max_w);

        let mut dist = math::distribute(config.n, p.as_slice().unwrap());
        dist.push(env.outputs); // Add output type

        let labels = csa::op::label(dist, config.model.k+1);

        // Weights
        //let l = labels.clone();
        //let w = ValueSet { f: Arc::new(
        //    move |i, j| t_w[[l(i) as usize, l(j) as usize]]
        //)};

        let w = weights(config.model.max_w);

        let dynamics = Self::default_dynamics();
        //let dynamics = Self::get_dynamics(m3, v2, labels.clone(), config);

        let sbm_mask = csa::op::sbm(labels.clone(), ValueSet::from_value(t_cpm.clone()));

        // geometric setup
        let coords: CoordinateFn = csa::op::random_coordinates(
            0.0,
            config.model.max_coordinate,
            config.n + env.outputs);

        let d: Metric = csa::op::distance_metric(coords.clone(), coords.clone());

        let disc = csa::op::disc(config.model.distance_threshold, d);

        let mask = sbm_mask & disc;

        let ns = NetworkSet {
            m: mask,
            v: vec![w],
            d: vec![dynamics]
        };

        let input_cs = Self::get_input_cs(v3, labels.clone(), coords.clone(), config);
        //let output_mask = Self::get_output_mask(v4, labels, coords.clone(), config);

        (ns, input_cs)
    }
}

fn weights(w: f32) -> ValueSet {
    ValueSet { f: Arc::new(
        move |_i, _j| w
    )}
}

impl MinimalModel {
    pub fn parse_params<'a>(ps: &'a ParameterSet, config: &'a RSNNConfig<Self>)
        -> (
            &'a Array2<f32>,
            &'a Array1<f32>,
            &'a Array1<f32>,
            ) {

        let t_cpm = match &ps.set[0] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        let p = match &ps.set[1] {
            Parameter::Vector(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        let input_t_cp = match &ps.set[2] {
            Parameter::Vector(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        assert!(t_cpm.shape() == [config.model.k+1, config.model.k+1]);
        assert!(p.shape() == [config.model.k]);
        assert!(input_t_cp.shape() == [config.model.k]);

        (t_cpm, p, input_t_cp)
    }

    fn get_dynamics(m: &Array2<f32>, v: &Array1<f32>, l: LabelFn, _config: &RSNNConfig<Self>) -> NeuronSet {
        let mut dm = m.mapv(|x| math::ml::sigmoid(x));

        let r = IzhikevichParameters::RANGES;

        for mut vals in dm.rows_mut() {
            vals[0] = vals[0] * (r[0].1 - r[0].0) + r[0].0;
            vals[1] = vals[1] * (r[1].1 - r[1].0) + r[1].0;
            vals[2] = vals[2] * (r[2].1 - r[2].0) + r[2].0;
            vals[3] = vals[3] * (r[3].1 - r[3].0) + r[3].0;

            vals[4] = if vals[4] > INHIBITORY_THRESHOLD {1.0} else {0.0};
        }

        let mut output_dynamics: Array1<f32> = Array::zeros(5);

        output_dynamics[0] = math::ml::sigmoid(v[0]) * (r[0].1 - r[0].0) + r[0].0;
        output_dynamics[1] = math::ml::sigmoid(v[1]) * (r[1].1 - r[1].0) + r[1].0;
        output_dynamics[2] = math::ml::sigmoid(v[2]) * (r[2].1 - r[2].0) + r[2].0;
        output_dynamics[3] = math::ml::sigmoid(v[3]) * (r[3].1 - r[3].0) + r[3].0;
        output_dynamics[4] = 0.0;

        let _ = dm.push_row(output_dynamics.view());

        NeuronSet { f: Arc::new(
            move |i| dm.slice(s![l(i) as usize, ..]).to_owned()
        )}
    }

    fn get_input_cs(v3: &Array1<f32>, l: LabelFn, g: CoordinateFn, config: &RSNNConfig<Self>)
        -> ConnectionSet {
        let input_t_cp = v3.mapv(|x| math::ml::sigmoid(x));
        //let input_t_w = v4.mapv(|x| math::ml::sigmoid(x) * config.model.max_w);

        let cp = ValueSet { f: Arc::new(
            move |i, _| input_t_cp[i as usize]
            )};

        let l2 = l.clone();
        let m = Mask {
            f: Arc::new(
                   move |i, j| random::random_range((0.0, 1.0)) < (cp.f)(l2(i), j)
               )
        };

        let input_g: CoordinateFn = Arc::new(move |_| (0.0, 0.0));

        let d: Metric = csa::op::distance_metric(g, input_g); // NOTE: Ensure this is correct

        let input_mask = m & csa::op::disc(config.model.distance_threshold, d);

        //let w = ValueSet { f: Arc::new(
        //    move |i, _| input_t_w[[l(i) as usize]]
        //)};

        // Default weights
        //let w = Self::default_weights();
        let w = weights(config.model.max_w);

        ConnectionSet {
            m: input_mask,
            v: vec![w]
        }
    }
}

impl Configurable for MinimalModel {
    type Config = MinimalModelConfig;
}

#[derive(Clone, Debug, Deserialize)]
pub struct MinimalModelConfig {
    pub k: usize,
    pub max_w: f32,
    pub distance_threshold: f32,
    pub max_coordinate: f32
}

impl ConfigSection for MinimalModelConfig {
    fn name() -> String {
        "test_model".to_string()
    }
}
