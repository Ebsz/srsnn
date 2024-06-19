use crate::models::Model;

use tasks::{Task, TaskEval};

use evolution::population::Individual;

use model::network::representation::{NetworkRepresentation, DefaultRepresentation};

use ndarray::{s, Axis, Array, Array1, Array2};

use std::fmt;

use petgraph::matrix_graph::MatrixGraph;
use petgraph::algo::astar;


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
            rank: item.n,
            size: item.edges().len(),

            matrix: item.connection_mask.clone()
        }
    }
}

impl Into<MatrixGraph<(), ()>> for &Graph {
    fn into(self) -> MatrixGraph<(),()> {
        let mut edges = Vec::new();

        for (i, d) in self.matrix.iter().enumerate() {
            let x = i / self.rank;
            let y = i % self.rank;

            if *d == 1 {
                edges.push((x as u16, y as u16));
            }
        }

        MatrixGraph::<(), ()>::from_edges(edges)
    }
}

pub struct GraphAnalysis {
    pub rank: usize,
    pub size: usize,

    pub avg_degree: f32,
    pub degree_dist: Array1<u32>, // Number of edges for each vertice

    pub small_world_coefficient: f32,
    pub clustering_coefficient: f32,
    pub avg_path_len: f32,
}

impl GraphAnalysis {
    pub fn analyze(g: &Graph) -> GraphAnalysis {
        let h = Self::avg_clustering(g);
        let l = Self::avg_path_length(g);

        let degree_dist = Self::degree_distribution(g);

        let avg_degree = degree_dist.iter().sum::<u32>() as f32 / g.rank as f32;

        GraphAnalysis {
            rank: g.rank,
            size: g.size,

            avg_degree,
            degree_dist,

            small_world_coefficient: h / l,
            clustering_coefficient: h,
            avg_path_len: l
        }
    }

    /// Calculate the small-world coefficient
    pub fn small_world(g: &Graph) -> f32 {
        let h = Self::avg_clustering(g);
        let l = Self::avg_path_length(g);

        h / l
    }

    fn degree_distribution(g: &Graph) -> Array1<u32> {
        g.matrix.sum_axis(Axis(1))
    }

    fn avg_path_length(g: &Graph) -> f32 {
        let mg: MatrixGraph<(),()> = g.into();

        let mut dst = 0;

        let mut n = 0;

        for i in 0..g.rank {
            for j in (i+1)..g.rank {
                let a = Self::path_length(&mg, i as u16, j as u16);
                let b = Self::path_length(&mg, j as u16, i as u16);

                dst += a + b;
            }
        }

        dst as f32 / (g.rank * (g.rank -1)) as f32
    }

    /// Calculate the shortest path from vertice s to vertice j,
    /// returning 0 if no such path exists.
    fn path_length(mg: &MatrixGraph<(),()>, s: u16, j: u16) -> u32 {
        let path = astar(&mg, s.into(), |finish| finish == j.into(), |e| 1, |_| 0);

        if let Some(res) = path {
            return res.0;
        }

        0
    }

    fn avg_clustering(g: &Graph) -> f32 {
        let sum_h: f32 = (0..g.rank).map(|i| Self::clustering(g, i)).sum();

        sum_h / g.rank as f32
    }

    /// Single vertice clustering coefficient
    fn clustering(g: &Graph, i: usize ) -> f32 {
        let neighbors = Self::neighborhood(g, i);

        let mut edges = 0;
        for i in 0..neighbors.len() {
            for j in (i+1)..neighbors.len() {
                if g.matrix[[i, j]] == 1 {
                    edges += 1;
                }
                if g.matrix[[j, i]] == 1 {
                    edges += 1
                }
            }
        }

        if edges == 0 {
            return 0.0;
        }

        let e_max = neighbors.len() * (neighbors.len() -1);
        let h = e_max as f32 / edges as f32;

        h
    }

    /// Vertices with edges to/from i
    fn neighborhood(g: &Graph, i: usize) -> Vec<usize>{
        let mut n: Vec<usize> = vec![];

        n.extend(g.matrix.slice(s![i,..]).iter().enumerate()
            .filter_map(|(n, x)| if *x == 1 && n != i {Some(n)} else { None }));

        n.extend(g.matrix.slice(s![..,i]).iter().enumerate()
            .filter_map(|(n, x)| if *x == 1 && n != i {Some(n)} else { None }));

        n.sort();
        n.dedup_by(|a, b| a == b);

        n
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
