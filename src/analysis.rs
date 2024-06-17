use crate::models::Model;

use tasks::{Task, TaskEval};

use evolution::population::Individual;

use model::network::representation::{NetworkRepresentation, DefaultRepresentation};

use ndarray::{s, Axis, Array, Array1, Array2};

use std::fmt;


pub struct Graph {
    pub rank: usize,
    pub size: usize,

    pub matrix: Array2<u32>
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "G(V={}, E={})", self.rank, self.size)
    }
}

impl<N> From<&NetworkRepresentation<N>> for Graph {
    fn from(item: &NetworkRepresentation<N>) -> Graph {
        Graph {
            rank: item.edges().len(),
            size: item.n,

            matrix: item.connection_mask.clone()
        }
    }
}

pub struct GraphAnalysis {
    pub rank: usize,
    pub size: usize,

    pub dg_dist: Array1<u32>, // Number of edges for each vertice
}

impl GraphAnalysis {
    pub fn analyze(g: &Graph) -> GraphAnalysis {

        GraphAnalysis {
            dg_dist: Self::degree_distribution(g),
            rank: g.rank,
            size: g.size
        }
    }

    fn degree_distribution(g: &Graph) -> Array1<u32> {
        g.matrix.sum_axis(Axis(1))
    }

    /// Calculate the small-world coefficient \lambda = \frac{H}{L}
    fn small_world(g: &Graph) {

    }
}

mod spikedata {
    use super::*;

    pub type SpikeSeries = Array2<f32>;

    pub fn to_firing_rate(data: SpikeSeries) -> Array2<f32> {
        const TIME_WINDOW: usize = 10; // number of timesteps we average over

        let max_t = data.nrows();

        let mut fr: Array2<f32> = Array::zeros(data.raw_dim());

        for t in 0..max_t {
            let t2 = std::cmp::min(t+TIME_WINDOW, max_t) as usize;

            let d: Array1<f32> = data.slice(s!(t..t2, ..)).sum_axis(Axis(0)).mapv(|x| x as f32 / max_t as f32) * 1000.0;

            fr.slice_mut(s!(t,..)).assign(&d);
        }

        fr
    }
}
