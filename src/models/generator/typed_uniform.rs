//! Typed Uniform Model
//! Connectivity and dynamics are set according to type, and is otherwise identical to the uniform

use crate::models::generator;
use crate::models::generator::config::TypedConfig;
use crate::models::generator::{Generator, NetworkModel};
use crate::models::generator_model::ModelConfig;

use csa::op::LabelFn;
use csa::{NetworkSet, ConnectionSet, ValueSet, NeuronSet, NeuronMask};
use csa::mask::Mask;

use model::neuron::izhikevich::IzhikevichParameters;

use utils::math;
use utils::config::{ConfigSection, Configurable, EmptyConfig};
use utils::environment::Environment;
use utils::parameters::{Parameter, ParameterSet};

use rand::Rng;
use serde::Deserialize;
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use std::sync::Arc;


const MAX_W: f32 = 5.0;

#[derive(Clone, Debug)]
pub struct TypedUniformModel;

impl Generator for TypedUniformModel {
    fn get(p: &ParameterSet, config: &ModelConfig<Self>, env: &Environment) -> (NetworkSet, ConnectionSet) {
        // Total number of types include the output types.
        let k = config.model.k + config.model.k_out;

        let p = vec![1.0/config.model.k as f32; config.model.k];
        let mut dist = math::distribute(config.n, &p);

        for _ in 0..config.model.k_out {
            dist.push(env.outputs / config.model.k_out);
        }

        let l = csa::op::label(dist, k);

        let mut rng = rand::thread_rng();

        // Generate random matrix of type-type connection proabilities.
        let t_cpm = Array::random((k,k), Uniform::new(0.0,1.0));
        let m = csa::op::sbm(l.clone(), ValueSet::from_value(t_cpm));

        //let p = rng.gen_range(0.0..1.0);
        //let m = csa::mask::random(p);

        let td = generator::blk::dynamics::uniform_typed(k);

        let d = csa::op::n_group(l.clone(), td);

        let w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };

        let ns = NetworkSet {
            m,
            v: vec![w],
            d: vec![d]
        };

        let input_t_cpm = Array::random((k, config.model.k_in), Uniform::new(0.0,1.0));

        let dist = vec![env.inputs / config.model.k_in;config.model.k_in];

        let input_l = csa::op::label(dist, config.model.k_in);

        let input_m = Mask {
            f: Arc::new(
               move |i, j| {
                   rand::thread_rng().gen_range(0.0..1.0) < input_t_cpm[[l(i) as usize, input_l(j) as usize]]
               })
        };

        let input_w = ValueSet { f: Arc::new( move |_i, _j| rand::thread_rng().gen_range(0.0..MAX_W) ) };
        let input_cs = ConnectionSet {
            m: input_m,
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

impl Configurable for TypedUniformModel {
    type Config = TypedConfig;
}
