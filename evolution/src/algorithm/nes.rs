use crate::algorithm::Algorithm;

use model::Model;

use utils::parameters::ParameterSet;
use utils::random;
use utils::config::{Configurable, ConfigSection};

use ndarray::{s, Array, Array1, Array2};
use ndarray_rand::rand_distr::StandardNormal;

use serde::Deserialize;


pub struct NES {
    w: Array1<f32>,
    noise_samples: Array2<f32>,

    population: Vec<ParameterSet>,

    param_structure: ParameterSet,

    conf: <Self as Configurable>::Config,
}

impl<M: Model> Algorithm<M> for NES {
    fn new(conf: Self::Config, m_conf: &M::Config) -> Self {
        let params = M::params(m_conf);
        let n_params = params.size();

        let w = random::random_vector(n_params, StandardNormal);

        let noise_samples = random::random_matrix((conf.population_size, n_params), StandardNormal);

        let population = Self::get_population(w.clone(), noise_samples.clone(), &conf, &params);

        log::info!("n_params: {n_params}");
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
        let std_evals = (&e - e.mean().unwrap()) / e.std(0.0);

        // update w
        self.w = &self.w + self.conf.alpha / (self.conf.population_size as f32 * self.conf.sigma)
            * self.noise_samples.t().dot(&std_evals);

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
    population_size: usize,
    alpha: f32,             // learning rate
    sigma: f32,             // noise stddev
}
impl ConfigSection for NESConfig {
    fn name() -> String {
        "NES".to_string()
    }
}
