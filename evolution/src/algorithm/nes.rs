/// Natural Evolution Strategies

use crate::algorithm::Algorithm;

use model::Model;

use utils::parameters::ParameterSet;
use utils::random;
use utils::config::{Configurable, ConfigSection};

use ndarray::{s, Array, Array1, Array2};
use ndarray_rand::rand_distr::{Normal, StandardNormal};

use serde::Deserialize;


pub struct NES {
    w: Array1<f32>,
    noise_samples: Array2<f32>,

    population: Vec<ParameterSet>,

    param_structure: ParameterSet,

    conf: <Self as Configurable>::Config,
}

impl Algorithm for NES {
    fn new<M: Model>(conf: Self::Config, m_conf: &M::Config) -> Self {
        let params = M::params(m_conf);
        let n_params = params.size();

        let w = random::random_vector(n_params, Normal::new(conf.init_mean,conf.init_stddev).unwrap());

        let noise_samples = random::random_matrix((conf.population_size, n_params), StandardNormal);

        let population = Self::get_population(w.clone(), noise_samples.clone(), &conf, &params);

        NES {
            w,
            noise_samples,
            population,
            param_structure: params,
            conf
        }
    }

    fn step(&mut self, evals: Vec<f32>) {
        let e = Array::from_vec(evals);

        if e.std(0.0) == 0.0 {
           log::warn!("eval stddev was 0.0, evals: {:#?}", e);

           log::warn!("skipping step");
           return;
        }

        let std_evals = (&e - e.mean().unwrap()) / e.std(0.0);

        // update w
        self.w = &self.w + self.conf.alpha / (self.conf.population_size as f32 * self.conf.sigma)
            * self.noise_samples.t().dot(&std_evals);

        assert!(self.w.iter().all(|x| !x.is_nan()),
            "updated w contained NaN: {:#?}\n\
             noise_samples: \n {:#?}\n\
             evals: \n {:#?}\n\
             e.std:: \n {:#?}\n\
             std_evals: \n {:#?}", self.w, self.noise_samples, &e, e.std(0.0), std_evals);

        // get new noise samples
        self.noise_samples = random::random_matrix((self.conf.population_size, self.w.len()), StandardNormal);

        // Update population
        self.population = Self::get_population(
            self.w.clone(),
            self.noise_samples.clone(),
            &self.conf,
            &self.param_structure
            );
    }

    fn parameter_sets(&self) ->&[ParameterSet] {
        for i in 0..self.population.len() {
            assert!(!self.population[i].is_nan(), "Error in parameter set: {:#?}", self.population[i].set);
        }

        &self.population
    }
}

impl NES {
    fn get_population(
        w: Array1<f32>,
        noise_samples: Array2<f32>,
        conf: &<Self as Configurable>::Config,
        param_structure: &ParameterSet
        ) -> Vec<ParameterSet> {
        let mut population = vec![];
        for i in 0..conf.population_size {
            let w_r = &w + conf.sigma * &noise_samples.slice(s![i,..]);

            let p = param_structure.assign(&w_r);
            population.push(p);
        }

        population
    }
}

impl Configurable for NES {
    type Config = NESConfig;
}

#[derive(Clone, Debug, Deserialize)]
pub struct NESConfig {
    pub population_size: usize,
    pub alpha: f32,             // learning rate
    pub sigma: f32,             // noise stddev
    pub init_mean: f32,
    pub init_stddev: f32,
}
impl ConfigSection for NESConfig {
    fn name() -> String {
        "NES".to_string()
    }
}
