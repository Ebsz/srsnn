use crate::algorithm::Algorithm;

use model::Model;

use utils::random;
use utils::parameters::ParameterSet;
use utils::config::{Configurable, ConfigSection};
use utils::environment::Environment;

use ndarray::{s, Array, Array1, Array2};
use ndarray_rand::rand_distr::{Normal, StandardNormal};

use serde::Deserialize;


pub struct SeparableNES {
    population: Vec<ParameterSet>,

    mu:    Array1<f32>, // means
    sigma: Array1<f32>, // variances
    s:     Array2<f32>, // noise samples

    n_params: usize,
    param_structure: ParameterSet,
    conf: SNESConfig
}

impl Algorithm for SeparableNES {
    fn new<M: Model>(conf: Self::Config, m_conf: &M::Config, env: &Environment) -> Self {
        let params = M::params(m_conf, env);
        let n_params = params.size();

        // Initialize means ~ N(0,1) and variances to 1.0, as per PS paper
        //let sigma: Array1<f32> = Array::ones(n_params);
        //let mu = random::random_vector(n_params, StandardNormal);

        let sigma: Array1<f32> = Array::ones(n_params) * conf.init_sigma;

        let mu = random::random_vector(n_params,
            Normal::new(conf.init_mu_mean, conf.init_mu_stddev).unwrap());

        let s = Self::get_s(&conf, n_params);

        SeparableNES {
            population: Self::get_population(&mu, &sigma, &s, &conf, &params),

            sigma,
            mu,

            s,

            param_structure: params,
            n_params,
            conf
        }
    }

    // TODO: Add indices to evals to ensure they are correct.
    fn step(&mut self, evals: Vec<f32>) {
        let e: Array1<f32> = Array::from_vec(evals);

        if e.std(0.0) == 0.0 {
           log::error!("eval stddev was 0.0 (should have been caught earlier)");

           self.reset();
           return;
        }

        // Normalize fitness to N(0, 1)
        let std_evals = (&e - e.mean().unwrap()) / e.std(0.0);

        // Compute gradients
        let g_mu    = &self.s.t().dot(&std_evals);
        let g_sigma = (self.s.mapv(|x| x.powf(2.0)) - 1.0).t().dot(&std_evals);

        // Update parameters
        self.mu = &self.mu + self.conf.lr_mu * &self.sigma * g_mu;
        self.sigma = &self.sigma * (self.conf.lr_sigma * 0.5 * g_sigma).mapv(f32::exp);

        self.s = Self::get_s(&self.conf, self.n_params);

        self.population = Self::get_population(
            &self.mu, &self.sigma, &self.s,
            &self.conf, &self.param_structure);
    }

    fn parameter_sets(&self) ->&[ParameterSet] {
        &self.population
    }
}

impl SeparableNES {
    fn get_population(
        mu: &Array1<f32>,
        sigma: &Array1<f32>,
        s: &Array2<f32>,
        conf: &SNESConfig,
        param_structure: &ParameterSet
    ) -> Vec<ParameterSet> {

        let mut population = vec![];

        // sample population
        for i in 0..conf.pop_size {
            let a = mu + sigma * &s.slice(s![i,..]);

            let p = param_structure.assign(&a);
            population.push(p);
        }

        population
    }

    fn get_s(conf: &SNESConfig, n_params: usize) -> Array2<f32> {
        random::random_matrix((conf.pop_size, n_params), StandardNormal)
    }

    fn reset(&mut self) {
        log::info!("Reinitializing SNES");
        let n_params = self.param_structure.size();

        self.sigma = Array::ones(n_params);
        self.mu = random::random_vector(n_params, StandardNormal);

        self.s = Self::get_s(&self.conf, n_params);

        self.population = Self::get_population(
            &self.mu, &self.sigma, &self.s,
            &self.conf, &self.param_structure);
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct SNESConfig {
    pub pop_size: usize,
    pub lr_mu: f32,    // \mu learning rate -  PS: 1.0
    pub lr_sigma: f32, // \sigma learning rate - PS: 0.01
    pub init_sigma: f32,

    pub init_mu_mean: f32,
    pub init_mu_stddev: f32
}

impl Configurable for SeparableNES {
    type Config = SNESConfig;
}

impl ConfigSection for SNESConfig {
    fn name() -> String {
        "snes".to_string()
    }
}
