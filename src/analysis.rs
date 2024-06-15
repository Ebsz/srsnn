use crate::models::Model;

use tasks::{Task, TaskEval};

use evolution::population::Individual;

use model::network::representation::DefaultRepresentation;

use ndarray::{s, Axis, Array, Array1, Array2};

use model::network::representation::NetworkRepresentation;


struct Graph {
    adjacency: Array2<u32>,
}

impl Graph {
    fn rank(&self) -> usize {
        self.adjacency.shape()[0]
    }
}

// TODO: Make this usable
pub struct NetworkAnalysis {
    pub rank: usize,
    pub size: usize,

    pub degree_distribution: Array1<u32>
}

pub fn analyze_individual<M: Model, T: Task + TaskEval>(i: &Individual<M, DefaultRepresentation>) {
    let desc = i.phenotype.as_ref().unwrap();

    let n = desc.n;

    println!("Resulting genome:");
    println!("n: {:?}", n);

    let density = desc.edges().len() as f32 / (n * n) as f32;

    println!("connection density {:?}", density);
}

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
