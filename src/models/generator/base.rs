//! Base generator model that other models are derived from, while also defining the baseline
//! of comparison for every other models.

use crate::models::generator::Generator;
use crate::models::generator_model::ModelConfig;

use csa::op::LabelFn;
use csa::op::geometric::{CoordinateFn, Metric};
use csa::{ConnectionSet, ValueSet, NeuronSet, NetworkSet};
use csa::mask::Mask;

use utils::{math, random};
use utils::config::{ConfigSection, Configurable};
use utils::parameters::{Parameter, ParameterSet};
use utils::environment::Environment;

use serde::Deserialize;

use ndarray::{array, Array, Array1, Array2};
use ndarray_rand::rand_distr::{Binomial, Distribution};

use std::sync::Arc;


// with N(0,1), 0.7 gives a starting inhibitory prob of ~0.2
const INHIBITORY_THRESHOLD: f32 = 0.70;
const INHIBITORY_FRACTION: f32 = 0.2;

const EXCITATORY_PARAMS: [f32; 5] = [0.02, 0.2, -65.0, 8.0, 0.0]; // Regular spiking (RS) neuron
const INHIBITORY_PARAMS: [f32; 5] = [0.1,  0.2, -65.0, 2.0, 1.0]; // Fast spiking (FS) neuron

const WT_DRAW_COUNT: u64 = 8;
const WT_DRAW_PROB: f64 = 0.2;


#[derive(Clone, Debug)]
pub struct BaseModel;

impl Generator for BaseModel {
    fn params(config: &ModelConfig<Self>, env: &Environment) -> ParameterSet {
        assert!(env.inputs % config.model.k_in == 0,
            "Number of input neurons({}) % input types({}) != 0", env.inputs, config.model.k_in);
        assert!(env.outputs % config.model.k_out == 0,
            "Number of output neurons({}) % output types({}) != 0", env.outputs, config.model.k_out);

        // Type connection probability matrix: [k + output]
        let t_cpm = Parameter::Matrix(Array::zeros(
                (config.model.k + config.model.k_out, config.model.k + config.model.k_out)));

        // Input->type cpm
        let input_t_cpm = Parameter::Matrix(Array::zeros((config.model.k, config.model.k_in)));

        ParameterSet {
            set: vec![t_cpm, input_t_cpm],
        }
    }

    fn get(
        params: &ParameterSet,
        config: &ModelConfig<Self>,
        env: &Environment)
        -> (NetworkSet, ConnectionSet)
    {
        let (m1, m2) = Self::parse_params(params, config);

        let t_cpm = m1.mapv(|x| math::ml::sigmoid(x));

        let p = vec![1.0/config.model.k as f32; config.model.k];
        let mut dist = math::distribute(config.n, &p);

        // Add output types
        for _ in 0..config.model.k_out {
            dist.push(env.outputs / config.model.k_out);
        }

        let labels = csa::op::label(dist, config.model.k+ config.model.k_out);

        let (dynamics, itypes) = Self::static_dynamics(labels.clone(), config);

        let sbm_mask = csa::op::sbm(labels.clone(), ValueSet::from_value(t_cpm.clone()));

        // geometric setup
        let coords: CoordinateFn = csa::op::geometric::random_coordinates(
            0.0,
            config.model.max_coordinate,
            config.n + env.outputs);

        let d: Metric = csa::op::geometric::distance_metric(coords.clone(), coords.clone());

        let disc = csa::op::geometric::disc(config.model.distance_threshold, d);

        let mask = sbm_mask & disc;

        let w = Self::minimal_weights(itypes, labels.clone(), config);

        let ns = NetworkSet {
            m: mask,
            v: vec![w],
            d: vec![dynamics]
        };

        let input_cs = Self::input_cs(m2, labels.clone(), coords.clone(), config, env);

        (ns, input_cs)
    }
}

impl BaseModel {
    pub fn parse_params<'a>(ps: &'a ParameterSet, config: &'a ModelConfig<Self>)
        -> (
            &'a Array2<f32>,
            &'a Array2<f32>,
            ) {

        let t_cpm = match &ps.set[0] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        let input_t_cpm = match &ps.set[1] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        assert!(t_cpm.shape() == [config.model.k + config.model.k_out, config.model.k + config.model.k_out]);
        assert!(input_t_cpm.shape() == [config.model.k, config.model.k_in]);

        (t_cpm, input_t_cpm)
    }

    fn minimal_weights(itypes: Vec<usize>, l: LabelFn, config: &ModelConfig<Self>) -> ValueSet {
        let w_map: Array1<f32> = (0..(config.model.k + config.model.k_out))
            .map(|i| if itypes.contains(&i) { config.model.inh_w } else { config.model.exc_w }).collect();

        let bin = Binomial::new(WT_DRAW_COUNT, WT_DRAW_PROB).unwrap();
        ValueSet { f: Arc::new(
            move |_i, j| w_map[l(j) as usize] * (1 + bin.sample(&mut rand::thread_rng())) as f32
        )}
    }

    fn static_dynamics(l: LabelFn, config: &ModelConfig<Self>) -> (NeuronSet, Vec<usize>) {
        let (td, itypes) = Self::type_dynamics(config);

        let dynamics_ns = csa::op::n_group(l, td);

        (dynamics_ns, itypes)
    }

    fn type_dynamics(config: &ModelConfig<Self>) -> (NeuronSet, Vec<usize>) {
        let d: Array2<f32> = array![EXCITATORY_PARAMS, INHIBITORY_PARAMS];

        let n_inhibitory_types = (config.model.k as f32 * INHIBITORY_FRACTION) as usize;

        let mut dist_map = vec![];

        dist_map.append(&mut vec![0; config.model.k - n_inhibitory_types]);
        dist_map.append(&mut vec![1; n_inhibitory_types]);
        dist_map.append(&mut vec![0; config.model.k_out]);

        let itypes: Vec<usize> = ((config.model.k-n_inhibitory_types)..config.model.k).collect();

        // Mapping from type indices -> dynamics indices
        let l: LabelFn = Arc::new(move |i| dist_map[i as usize] as u32);

        let v_d = NeuronSet::from_value(d);

        (csa::op::n_group(l, v_d), itypes)
    }

    fn input_cs(m2: &Array2<f32>, l: LabelFn, g: CoordinateFn, config: &ModelConfig<Self>, env: &Environment)
        -> ConnectionSet {
        let input_t_cpm = m2.mapv(|x| math::ml::sigmoid(x));

        // Distribute the input neurons equally over each input type
        let dist = vec![env.inputs / config.model.k_in;config.model.k_in];

        let input_label = csa::op::label(dist, config.model.k_in);

        let m = Mask {
            f: Arc::new(
               move |i, j| {
                   random::random_range((0.0, 1.0)) < input_t_cpm[[l(i) as usize, input_label(j) as usize]]
               })
        };

        let max_x = config.model.max_coordinate;
        let g_in: CoordinateFn = Arc::new(move |_| (rand::random::<f32>() * max_x, 0.0));
        let d: Metric = csa::op::geometric::distance_metric(g, g_in);

        let input_mask = m & csa::op::geometric::disc(config.model.distance_threshold, d);

        let w = weights(config.model.input_w);

        ConnectionSet {
            m: input_mask,
            v: vec![w]
        }
    }
}

fn weights(w: f32) -> ValueSet {
    let bin = Binomial::new(WT_DRAW_COUNT, WT_DRAW_PROB).unwrap();
    ValueSet { f: Arc::new(
        move |_i, _j| w * (1 + bin.sample(&mut rand::thread_rng())) as f32
    )}
}

impl Configurable for BaseModel {
    type Config = BaseModelConfig;
}

#[derive(Clone, Debug, Deserialize)]
pub struct BaseModelConfig {
    pub k: usize,       // # of types

    pub k_in: usize,    // # of input types
    pub k_out: usize,   // # of output types

    pub distance_threshold: f32,
    pub max_coordinate: f32,

    pub input_w: f32,
    pub exc_w: f32,
    pub inh_w: f32,
}

impl ConfigSection for BaseModelConfig {
    fn name() -> String {
        "base_model".to_string()
    }
}
