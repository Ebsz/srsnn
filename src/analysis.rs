use crate::runnable::RunnableNetwork;

use model::DefaultNetwork;
use model::network::representation::{NetworkRepresentation, DefaultRepresentation};
use model::record::Record;

use tasks::{Task, TaskEval};
use tasks::task_runner::TaskRunner;

use ndarray::{s, Axis, Array, Array1, Array2};

use graph::{Graph, GraphAnalysis};


pub fn run_analysis<T: Task + TaskEval> (repr: &DefaultRepresentation, setups: &[T::Setup])
-> Vec<Record> {
    log::debug!("Performing run analysis");

    let mut records = vec![];

    for s in setups {
        let task = T::new(&setups[0]);
        let mut runnable = RunnableNetwork::<DefaultNetwork>::build(repr);
        runnable.network.enable_recording();

        let mut runner = TaskRunner::new(task, &mut runnable);
        runner.run();

        records.push(runnable.network.record);
    }

    records
}

pub fn analyze_network(r: &DefaultRepresentation) -> (Graph, GraphAnalysis) {
    let graph: Graph = r.into();
    let graph_analysis = GraphAnalysis::analyze(&graph);

    println!("graph analysis:");
    println!("{graph}\n\n");
    println!("{graph_analysis}");

    let n_inhibitory: f32 = r.neurons.iter()
        .map(|n| f32::from(n.inhibitory)).sum();

    println!("");
    println!("inhibitory: {} ({:.3}%)", n_inhibitory, (n_inhibitory / r.neurons.len() as f32) * 100.0);

    println!("weights - mean: {:.3}, std: {:.3}", r.network_w.mean().unwrap(), r.network_w.std(0.0));

    let n_input_connections: u32 = r.input_cm.iter().sum();
    let input_density = n_input_connections as f32 / (r.input_cm.shape()[0] * r.input_cm.shape()[1]) as f32;
    println!("Input - {n_input_connections} input connections, density: {input_density}");
    println!("Input weights - mean: {:.3}, std: {:.3}", r.input_w.mean().unwrap(), r.input_w.std(0.0));

    //let n_output_connections: u32 = r.output_cm.iter().sum();
    //let output_density = n_output_connections as f32 / (r.output_cm.shape()[0] * r.output_cm.shape()[1]) as f32;
    //println!("Output - {n_output_connections} output connections, density: {output_density}");

    (graph, graph_analysis)
}

pub mod graph {
    use super::*;

    use petgraph::matrix_graph::MatrixGraph;
    use petgraph::algo::astar;

    use petgraph::dot;
    use std::fmt;

    pub struct Graph {
        pub rank: usize,
        pub size: usize,

        pub matrix: Array2<u32>
    }

    impl Graph {
        /// Return a reduced Graph with isolated vertices removed
        pub fn reduce(g: &Graph) -> Graph {
            let i_vtc = GraphAnalysis::isolated_vertices(g);

            let rank = g.rank - i_vtc.len();

            let mut mx: Array2<u32> =  Array::zeros((g.rank, 0));
            for (i, c) in g.matrix.columns().into_iter().enumerate() {
                if !i_vtc.contains(&(i as u32)) {
                    let _ = mx.push_column(c);
                }
            }

            let mut matrix: Array2<u32> = Array::zeros((0, mx.shape()[1]));
            for (i, r) in mx.rows().into_iter().enumerate() {
                if !i_vtc.contains(&(i as u32)) {
                    let _ = matrix.push_row(r);
                }
            }

            Graph {
                rank,
                size: g.size,
                matrix,
            }
        }

        pub fn dot(&self) {
            let mg: petgraph::Graph<(), ()> = self.into();

            let dot = dot::Dot::with_config(&mg, &[dot::Config::EdgeNoLabel, dot::Config::NodeNoLabel]);
            println!("{:?}", dot);
        }

        pub fn edges(&self) -> Vec<(u32, u32)> {
            let mut edges = Vec::new();

            for (i, d) in self.matrix.iter().enumerate() {
                let x = i / self.rank;
                let y = i % self.rank;

                if *d == 1 {
                    edges.push((x as u32, y as u32));
                }
            }

            edges
        }
    }


    impl<N> From<&NetworkRepresentation<N>> for Graph {
        fn from(item: &NetworkRepresentation<N>) -> Graph {
            Graph {
                rank: item.n,
                size: item.edges().len(),

                matrix: item.network_cm.clone()
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

    impl Into<petgraph::Graph<(), ()>> for &Graph {
        fn into(self) -> petgraph::Graph<(), ()> {
            let mut edges = Vec::new();

            for (i, d) in self.matrix.iter().enumerate() {
                let x = i / self.rank;
                let y = i % self.rank;

                if *d == 1 {
                    edges.push((x as u32, y as u32));
                }
            }

            petgraph::Graph::<(), ()>::from_edges(&edges)
        }
    }

    impl fmt::Display for Graph {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "G(V={}, E={})", self.rank, self.size)
        }
    }

    pub struct GraphAnalysis {
        pub rank: usize,
        pub size: usize,

        pub density: f32,
        pub avg_degree: f32,
        pub degree_dist: Array1<u32>, // Number of edges for each vertice

        pub small_world_coefficient: f32,
        pub clustering_coefficient: f32,
        pub avg_path_len: f32,
    }

    impl fmt::Display for GraphAnalysis {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "rank: {}\n\
                size: {}\n\
                density: {}\n\n\
                average degree: {:.3}\n\
                small world: {:.3}\n\
                clustering: {}\n\
                average path length: {}",
                self.rank, self.size, self.density, self.avg_degree,
                self.small_world_coefficient, self.clustering_coefficient,
                self.avg_path_len)
        }
    }

    impl GraphAnalysis {
        pub fn analyze(g: &Graph) -> GraphAnalysis {
            let density = Self::density(g);

            let h = Self::avg_clustering(g);
            let l = Self::avg_path_length(g);

            let degree_dist = Self::degree_distribution(g);

            let avg_degree = degree_dist.iter().sum::<u32>() as f32 / g.rank as f32;

            GraphAnalysis {
                rank: g.rank,
                size: g.size,

                density,
                avg_degree,
                degree_dist,

                small_world_coefficient: h / l,
                clustering_coefficient: h,
                avg_path_len: l
            }
        }

        pub fn density(g: &Graph) -> f32 {
            g.size as f32 / (g.rank as f32).powf(2.0)
        }


        pub fn isolated_vertices(g: &Graph) -> Vec<u32> {
            let mut i_vtc = Vec::new();

            for v in 0..g.rank {
                if g.matrix.slice(s![v,..]).sum() + g.matrix.slice(s![..,v]).sum() == 0 {
                    i_vtc.push(v as u32);
                }
            }

            i_vtc
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
            let path = astar(&mg, s.into(), |finish| finish == j.into(), |_| 1, |_| 0);

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
}
