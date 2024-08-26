//! Runnable representation of a spiking network

use crate::neuron::NeuronModel;
use crate::neuron::izhikevich::Izhikevich;

use utils::environment::Environment;

use ndarray::{Array1, Array2};

use serde::{Serialize, Deserialize};



pub type DefaultRepresentation = NetworkRepresentation<NeuronDescription<Izhikevich>>;

#[derive(Clone, Deserialize, Serialize)]
pub struct NetworkRepresentation<N> {
    pub n: usize,
    pub neurons: Array1<N>,

    pub network_cm: Array2<u32>,
    pub network_w: Array2<f32>,

    pub input_cm: Array2<u32>,
    pub input_w: Array2<f32>,

    pub env: Environment,
}

impl<N> NetworkRepresentation<N> {
    pub fn new(
        neurons: Array1<N>,
        network_cm: Array2<u32>,
        network_w: Array2<f32>,

        input_cm: Array2<u32>,
        input_w: Array2<f32>,

        env: Environment)
        -> NetworkRepresentation<N>
    {

        // Ensure weights are non-negative
        assert!(network_w.iter().filter(|x| **x < 0.0).collect::<Vec<&f32>>().len() == 0,
            "network weight matrix contains negative entries");
        assert!(input_w.iter().filter(|x| **x < 0.0).collect::<Vec<&f32>>().len() == 0,
            "input weight matrix contains negative entries");

        let n = neurons.shape()[0];

        // |n| == |cm|
        assert!(network_cm.shape() == [n,n],
            "# neurons ({:?}) != network connection matrix: ({:?})", neurons.shape()[0], network_cm.shape());

        let n_rec = n - env.outputs;

        // |input_cm| == [n_rec x n_in]
        assert!(input_cm.shape() == [n_rec, env.inputs]);

        // |cm| == |w|
        assert!(network_cm.shape() == network_w.shape());
        assert!(input_cm.shape() == input_w.shape());

        NetworkRepresentation {
            n,
            neurons,
            network_cm,
            network_w,

            input_cm,
            input_w,

            env,
        }
    }

    pub fn edges(&self) -> Vec<(u32, u32)> {
        let mut edges = Vec::new();

        for (i, d) in self.network_cm.iter().enumerate() {
            let x = i / self.n;
            let y = i % self.n;

            if *d == 1 {
                edges.push((x as u32, y as u32));
            }
        }

        edges
    }
}

#[derive(Copy, Debug, Deserialize, Serialize)]
pub struct NeuronDescription<N: NeuronModel> {
    pub id: u32,
    pub params: N::Parameters,
    pub inhibitory: bool,
}

impl<N: NeuronModel> NeuronDescription<N> {
    pub fn new(id: u32, params: N::Parameters, inhibitory: bool) -> NeuronDescription<N> {
        NeuronDescription {
            id,
            params,
            inhibitory,
        }
    }
}

impl<N: NeuronModel> Clone for NeuronDescription<N> {
    fn clone(&self) -> NeuronDescription<N> {
        NeuronDescription {
            id: self.id,
            params: self.params,
            inhibitory: self.inhibitory,
        }
    }
}
