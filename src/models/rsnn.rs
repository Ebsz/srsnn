/// Generic model of a recurrent spiking neural network

use csa::{ConnectionSet, ValueSet, NeuronSet, NetworkSet};
use csa::mask::Mask;

use model::Model;
use model::network::representation::{DefaultRepresentation, NetworkRepresentation, NeuronDescription};
use model::neuron::izhikevich::IzhikevichParameters;

use utils::parameters::ParameterSet;
use utils::config::{Configurable, ConfigSection};
use utils::environment::Environment;

use serde::Deserialize;

use ndarray::{s, array};

use std::sync::Arc;
use std::fmt::Debug;


pub trait RSNN: Configurable + Clone + Debug + Sync {
    fn get(p: &ParameterSet, config: &RSNNConfig<Self>, env: &Environment) -> (NetworkSet, ConnectionSet);
    fn params(config: &RSNNConfig<Self>, env: &Environment) -> ParameterSet;

    fn default_dynamics() -> NeuronSet {
        NeuronSet { f: Arc::new(
            move |_i| array![0.02, 0.2, -65.0, 2.0, 0.0]
        )}
    }

    fn default_weights() -> ValueSet {
        ValueSet { f: Arc::new(
            move |_i, _j| 1.0
        )}
    }

    fn default_input(n: usize) -> ConnectionSet {
        let m_in = Mask { f: Arc::new( move |i, j| i == j ) };

        let w_in = Self::default_weights();

        ConnectionSet {
            m: m_in,
            v: vec![w_in]
        }
    }

    fn default_output() -> Mask {
        Mask { f: Arc::new( move |i, j| i == j ) }
    }

    /// 4 Izhikevich parameters + inhibitory flag
    const N_DYNAMICAL_PARAMETERS: usize = 5;
}

pub struct RSNNModel<R: RSNN> {
    n: usize,
    conf: RSNNConfig<R>,

    params: ParameterSet,
    env: Environment,
}

impl<R: RSNN> Model for RSNNModel<R> {
    fn new(
        conf: &RSNNConfig<R>,
        params: &ParameterSet,
        env: &Environment) -> Self {
        RSNNModel {
            n: conf.n,
            params: params.clone(),
            env: env.clone(),

            conf: conf.clone()
        }
    }

    fn develop(&self) -> DefaultRepresentation {
        let n = self.n + self.env.outputs;

        let (neural_set, input_cs) = R::get(&self.params, &self.conf, &self.env);

        let mask = neural_set.m;
        let dynamics = &neural_set.d[0];

        // Self connections are always removed
        let network_cm = (mask - csa::mask::one_to_one()).matrix(n);

        let d = dynamics.vec(n);

        let mut neurons = Vec::new();

        for i in 0..n {
            let inhibitory = if d[i][4] == 1.0 { true } else { false };
            neurons.push(NeuronDescription::new(
                    i as u32,
                    IzhikevichParameters {
                        a: d[i][0],
                        b: d[i][1],
                        c: d[i][2],
                        d: d[i][3],
                    },
                    inhibitory,
            ));
        }

        let network_w = neural_set.v[0].matrix(n);

        let input_cm = input_cs.m.r_matrix(self.n, self.env.inputs);

        let input_w = input_cs.v[0].r_matrix(self.n, self.env.inputs);

        //let output_cm = output_mask.r_matrix(self.env.outputs, self.n);

        //if output_cm.iter().all(|c| *c == 0) {
        //    log::trace!("No output connections");
        //}

        if input_cm.iter().all(|c| *c == 0) {
            log::trace!("No connections from input");
        }

        NetworkRepresentation::new(neurons.into(),
            network_cm,
            network_w,
            input_cm,
            input_w,
            self.env.clone())
    }

    fn params(config: &RSNNConfig<R>, env: &Environment) -> ParameterSet {
        R::params(config, env)
    }
}

impl<R: RSNN> Configurable for RSNNModel<R> {
    type Config = RSNNConfig<R>;
}

#[derive(Clone, Debug, Deserialize)]
pub struct RSNNConfig<R: RSNN> {
    pub n: usize,
    pub model: R::Config
}

impl<R: RSNN> ConfigSection for RSNNConfig<R> {
    fn name() -> String {
        "rsnn".to_string()
    }
}
