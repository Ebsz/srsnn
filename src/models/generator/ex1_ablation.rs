//! Set of Ablation models for Experiment 1

use crate::models::generator;
use crate::models::generator::{Generator, NetworkModel};
use crate::models::generator_model::ModelConfig;
use crate::models::generator::config::TypedConfig;

use csa::{NetworkSet, ConnectionSet, ValueSet, NeuronSet, NeuronMask};
use csa::mask::Mask;
use csa::op::LabelFn;
use csa::op::geometric::{CoordinateFn, Metric};

use model::neuron::izhikevich::IzhikevichParameters;

use utils::{math, random};
use utils::config::{ConfigSection, Configurable, EmptyConfig};
use utils::environment::Environment;
use utils::parameters::{Parameter, ParameterSet};

use rand::Rng;
use ndarray::{array, Array, Array2};

use serde::Deserialize;

use std::sync::Arc;


const MAX_W: f32 = 5.0;

#[derive(Clone, Debug)]
pub struct GeometricModel;

impl Generator for GeometricModel {
    fn get(p: &ParameterSet, config: &ModelConfig<Self>, env: &Environment) -> (NetworkSet, ConnectionSet) {
        let mut rng = rand::thread_rng();

        let p = rng.gen_range(0.0..1.0);
        let random_mask = csa::mask::random(p);

        // geometric setup
        let coords: CoordinateFn = csa::op::geometric::random_coordinates(
            0.0,
            10.0, // Max coordinate
            config.n + env.outputs);

        let d: Metric = csa::op::geometric::distance_metric(coords.clone(), coords.clone());
        let disc = csa::op::geometric::disc(3.0, d); // Distance threshold 

        let m = random_mask & disc;

        let d = generator::blk::dynamics::uniform();

        let w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let ns = NetworkSet {
            m,
            v: vec![w],
            d: vec![d]
        };

        let p = rng.gen_range(0.0..1.0);
        let input_mask = csa::mask::random(p);

        let input_w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let input_cs = ConnectionSet {
            m: input_mask,
            v: vec![input_w]
        };

        (ns, input_cs)
    }

    fn params(config: &ModelConfig<Self>, env: &Environment) -> ParameterSet {
        ParameterSet {
            set: vec![]
        }
    }
}

impl Configurable for GeometricModel {
    type Config = EmptyConfig;
}


/// Typed Model
#[derive(Clone, Debug)]
pub struct TypedModel;

impl Generator for TypedModel {
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

        let labels = csa::op::label(dist, config.model.k + config.model.k_out);

        let dynamics = generator::blk::dynamics::uniform();

        let mask = csa::op::sbm(labels.clone(), ValueSet::from_value(t_cpm.clone()));

        // Uniform weights
        let w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let ns = NetworkSet {
            m: mask,
            v: vec![w],
            d: vec![dynamics]
        };

        let input_cs = Self::input_cs(m2, labels.clone(), config, env);

        (ns, input_cs)
    }
}

impl TypedModel {
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

    fn input_cs(m2: &Array2<f32>, l: LabelFn, config: &ModelConfig<Self>, env: &Environment)
        -> ConnectionSet {
        let input_t_cpm = m2.mapv(|x| math::ml::sigmoid(x));

        // Distribute the input neurons equally over each input type
        let dist = vec![env.inputs / config.model.k_in;config.model.k_in];

        let input_label = csa::op::label(dist, config.model.k_in);

        let input_mask = Mask {
            f: Arc::new(
               move |i, j| {
                   random::random_range((0.0, 1.0)) < input_t_cpm[[l(i) as usize, input_label(j) as usize]]
               })
        };

        let w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        ConnectionSet {
            m: input_mask,
            v: vec![w]
        }
    }
}

impl Configurable for TypedModel {
    type Config = TypedConfig;
}

/// GeometricTyped Model
#[derive(Clone, Debug)]
pub struct GeometricTypedModel;

impl Generator for GeometricTypedModel {
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

        let labels = csa::op::label(dist, config.model.k + config.model.k_out);

        let dynamics = generator::blk::dynamics::uniform();
        
        // geometric setup
        let coords: CoordinateFn = csa::op::geometric::random_coordinates(
            0.0,
            10.0, // Max coordinate
            config.n + env.outputs);

        let d: Metric = csa::op::geometric::distance_metric(coords.clone(), coords.clone());
        let disc = csa::op::geometric::disc(3.0, d); // Distance threshold 

        let sbm_mask = csa::op::sbm(labels.clone(), ValueSet::from_value(t_cpm.clone()));
        
        let mask = sbm_mask & disc;

        // Uniform weights
        let w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let ns = NetworkSet {
            m: mask,
            v: vec![w],
            d: vec![dynamics]
        };

        let input_cs = Self::input_cs(m2, labels.clone(), coords.clone(), config, env);

        (ns, input_cs)
    }
}

impl GeometricTypedModel {
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
        
        let g_in: CoordinateFn = Arc::new(move |_| (rand::random::<f32>() * 10.0, 0.0));
        let d: Metric = csa::op::geometric::distance_metric(g, g_in);

        let input_mask = m & csa::op::geometric::disc(3.0, d);

        let w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        ConnectionSet {
            m: input_mask,
            v: vec![w]
        }
    }
}

impl Configurable for GeometricTypedModel {
    type Config = TypedConfig;
}
