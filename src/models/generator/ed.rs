//! Evolved-Dynamics model, which is identical to the base model except that it is additionally
//! parameterized by type-based dynamical parameters.
//!


use crate::models::generator::Generator;
use crate::models::generator_model::ModelConfig;

use model::neuron::izhikevich::IzhikevichParameters;

use csa::op::LabelFn;
use csa::op::geometric::{CoordinateFn, Metric};
use csa::{ConnectionSet, ValueSet, NeuronSet, NetworkSet};
use csa::mask::Mask;


use utils::{math, random};
use utils::config::{ConfigSection, Configurable};
use utils::parameters::{Parameter, ParameterSet};
use utils::environment::Environment;

use serde::Deserialize;

use ndarray::{s, array, Array, Array1, Array2};
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
pub struct EvolvedDynamicsModel;

impl Generator for EvolvedDynamicsModel {
    fn params(config: &ModelConfig<Self>, env: &Environment) -> ParameterSet {
        assert!(env.inputs % config.model.k_in == 0,
            "Number of input neurons({}) % input types({}) != 0", env.inputs, config.model.k_in);
        assert!(env.outputs % config.model.k_out == 0,
            "Number of output neurons({}) % output types({}) != 0", env.outputs, config.model.k_out);

        // Type connection probability matrix: [k + k_out, k + k_out]
        let t_cpm = Parameter::Matrix(Array::zeros(
                (config.model.k + config.model.k_out, config.model.k + config.model.k_out)));

        // Input->type cpm
        let input_t_cpm = Parameter::Matrix(Array::zeros((config.model.k, config.model.k_in)));

        // Dynamical parameters: [k, 5]
        let d = Parameter::Matrix(Array::zeros((config.model.k, Self::N_DYNAMICAL_PARAMETERS)));

        // Dynamical parameters for output type does not include inhibitory flag: [k_out, 4]
        let d_out = Parameter::Matrix(Array::zeros((config.model.k_out, Self::N_DYNAMICAL_PARAMETERS -1)));

        ParameterSet {
            set: vec![t_cpm, input_t_cpm, d, d_out],
        }
    }

    fn get(
        params: &ParameterSet,
        config: &ModelConfig<Self>,
        env: &Environment)
        -> (NetworkSet, ConnectionSet)
    {
        let (m1, m2, m3, m4) = Self::parse_params(params, config);

        let t_cpm = m1.mapv(|x| math::ml::sigmoid(x));

        let p = vec![1.0/config.model.k as f32; config.model.k];
        let mut dist = math::distribute(config.n, &p);

        // Add output types
        for _ in 0..config.model.k_out {
            dist.push(env.outputs / config.model.k_out);
        }

        let labels = csa::op::label(dist, config.model.k+ config.model.k_out);

        let (dynamics, itypes) = Self::evolved_dynamics(labels.clone(), m3, m4, config);

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

impl EvolvedDynamicsModel {
    pub fn parse_params<'a>(ps: &'a ParameterSet, config: &'a ModelConfig<Self>)
        -> (
            &'a Array2<f32>,
            &'a Array2<f32>,
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

        let d = match &ps.set[2] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        let d_out = match &ps.set[3] {
            Parameter::Matrix(x) => {x},
            _ => { panic!("invalid parameter set") }
        };

        assert!(t_cpm.shape() == [config.model.k + config.model.k_out, config.model.k + config.model.k_out]);
        assert!(input_t_cpm.shape() == [config.model.k, config.model.k_in]);
        assert!(d.shape() == [config.model.k, Self::N_DYNAMICAL_PARAMETERS]);
        assert!(d_out.shape() == [config.model.k_out, Self::N_DYNAMICAL_PARAMETERS-1]);

        (t_cpm, input_t_cpm, d, d_out)
    }

    fn minimal_weights(itypes: Vec<usize>, l: LabelFn, config: &ModelConfig<Self>) -> ValueSet {
        let w_map: Array1<f32> = (0..(config.model.k + config.model.k_out))
            .map(|i| if itypes.contains(&i) { config.model.inh_w } else { config.model.exc_w }).collect();

        let bin = Binomial::new(WT_DRAW_COUNT, WT_DRAW_PROB).unwrap();
        ValueSet { f: Arc::new(
            move |_i, j| w_map[l(j) as usize] * (1 + bin.sample(&mut rand::thread_rng())) as f32
        )}
    }

    fn evolved_dynamics(l: LabelFn, m3: &Array2<f32>, m4: &Array2<f32>, config: &ModelConfig<Self>) 
        -> (NeuronSet, Vec<usize>) {

        // Calculate the number of inhibitory types, ensuring minimum one inhibitory type.
        let n_inhibitory_types = ((config.model.k as f32 * INHIBITORY_FRACTION) as usize).max(1);

        let r = IzhikevichParameters::RANGES;

        // Recurrent neurons
        let mut dm = m3.mapv(|x| math::ml::sigmoid(x));

        let mut i = 0;
        for mut vals in dm.rows_mut() {
            vals[0] = vals[0] * (r[0].1 - r[0].0) + r[0].0;
            vals[1] = vals[1] * (r[1].1 - r[1].0) + r[1].0;
            vals[2] = vals[2] * (r[2].1 - r[2].0) + r[2].0;
            vals[3] = vals[3] * (r[3].1 - r[3].0) + r[3].0;

            //vals[4] = if vals[4] > INHIBITORY_THRESHOLD {1.0} else {0.0};
            vals[4] = if i >= config.model.k - n_inhibitory_types {
                1.0
            } else {
                0.0
            };

            i += 1;
        }

        // Output neurons
        let dmo = m4.mapv(|x| math::ml::sigmoid(x));

        for mut vals in dmo.rows() {
            let mut od = Array::zeros(5);
            od[0] = vals[0] * (r[0].1 - r[0].0) + r[0].0;
            od[1] = vals[1] * (r[1].1 - r[1].0) + r[1].0;
            od[2] = vals[2] * (r[2].1 - r[2].0) + r[2].0;
            od[3] = vals[3] * (r[3].1 - r[3].0) + r[3].0;

            od[4] = 0.0;
            dm.push_row(od.view());
        }

        let itypes: Vec<usize> = ((config.model.k-n_inhibitory_types)..config.model.k).collect();

        let v_d = NeuronSet::from_value(dm);

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

impl Configurable for EvolvedDynamicsModel {
    type Config = EDModelConfig;
}

#[derive(Clone, Debug, Deserialize)]
pub struct EDModelConfig {
    pub k: usize,       // # of types

    pub k_in: usize,    // # of input types
    pub k_out: usize,   // # of output types

    pub distance_threshold: f32,
    pub max_coordinate: f32,

    pub input_w: f32,
    pub exc_w: f32,
    pub inh_w: f32,
}

impl ConfigSection for EDModelConfig {
    fn name() -> String {
        "base_model".to_string()
    }
}
