//! Base model that other models are derived from, while also defining the baseline
//! of comparison for every other models.

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
use ndarray_rand::rand_distr::{Binomial, Distribution};

use std::sync::Arc;


// with N(0,1), 0.7 gives a starting inhibitory prob of ~0.2
const INHIBITORY_THRESHOLD: f32 = 0.70;
const INHIBITORY_FRACTION: f32 = 0.2;

const EXCITATORY_PARAMS: [f32; 5] = [0.02, 0.2, -65.0, 8.0, 0.0]; // Regular spiking (RS) neuron
const INHIBITORY_PARAMS: [f32; 5] = [0.1,  0.2, -65.0, 2.0, 1.0]; // Fast spiking (FS) neuron

const EXCITATORY_WT: f32 = 0.33;
const INHIBITORY_WT: f32 = 0.41;
const INPUT_WT: f32 = 0.55;

const WT_DRAW_COUNT: u64 = 8;
const WT_DRAW_PROB: f64 = 0.2;


#[derive(Clone, Debug)]
pub struct BaseModel;

impl RSNN for BaseModel {
    fn params(config: &RSNNConfig<Self>, env: &Environment) -> ParameterSet {
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
        config: &RSNNConfig<Self>,
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
        let coords: CoordinateFn = csa::op::random_coordinates(
            0.0,
            config.model.max_coordinate,
            config.n + env.outputs);

        let d: Metric = csa::op::distance_metric(coords.clone(), coords.clone());

        let disc = csa::op::disc(config.model.distance_threshold, d);

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
    pub fn parse_params<'a>(ps: &'a ParameterSet, config: &'a RSNNConfig<Self>)
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

    fn minimal_weights(itypes: Vec<usize>, l: LabelFn, config: &RSNNConfig<Self>) -> (ValueSet) {
        let w_map: Array1<f32> = (0..(config.model.k + config.model.k_out))
            .map(|i| if itypes.contains(&i) { INHIBITORY_WT} else {EXCITATORY_WT}).collect();

        let bin = Binomial::new(WT_DRAW_COUNT, WT_DRAW_PROB).unwrap();
        ValueSet { f: Arc::new(
            move |_i, j| w_map[l(j) as usize] * (1 + bin.sample(&mut rand::thread_rng())) as f32
        )}
    }

    fn static_dynamics(l: LabelFn, config: &RSNNConfig<Self>) -> (NeuronSet, Vec<usize>) {
        // Dynamics matrix: each row is the dynamical parameters for a type
        let mut dm: Array2<f32> = Array::zeros((config.model.k + config.model.k_out, 5));

        let inhibitory: Array1<f32> = INHIBITORY_PARAMS.into_iter().collect();
        let excitatory: Array1<f32> = EXCITATORY_PARAMS.into_iter().collect();

        let n_inhibitory_types = (config.model.k as f32 * INHIBITORY_FRACTION) as usize;

        let mut itypes = vec![];

        for i in 0..(config.model.k - n_inhibitory_types) {
            dm.slice_mut(s![i,..]).assign(&excitatory);
        }

        for i in (config.model.k - n_inhibitory_types)..config.model.k {
            dm.slice_mut(s![i,..]).assign(&inhibitory);
            itypes.push(i);
        }

        // Set output types to always be excitatory
        for i in 0..config.model.k_out {
            dm.slice_mut(s![-(1+ i as i32),..]).assign(&excitatory);
        }

        (NeuronSet { f: Arc::new(
            move |i| dm.slice(s![l(i) as usize, ..]).to_owned()
        )},
        itypes)
    }

    fn input_cs(m2: &Array2<f32>, l: LabelFn, g: CoordinateFn, config: &RSNNConfig<Self>, env: &Environment)
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
        let d: Metric = csa::op::distance_metric(g, g_in);

        let input_mask = m & csa::op::disc(config.model.distance_threshold, d);

        let w = weights(INPUT_WT);

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
    pub max_coordinate: f32
}

impl ConfigSection for BaseModelConfig {
    fn name() -> String {
        "base_model".to_string()
    }
}


