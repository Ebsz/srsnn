use luna::gen;
use luna::models::Model;
use luna::models::stochastic::StochasticModel;
use luna::models::stochastic::main_model::{MainStochasticModel, MainModelConfig};
use luna::visual::visualize_network;

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;

use model::network::representation::NetworkRepresentation;

use utils::logger::init_logger;

use ndarray::Array2;

use std::convert::From;
use std::fmt;


const N: usize = 100;

struct Graph {
    rank: usize,
    size: usize,

    matrix: Array2<u32>
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

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "G(V={}, E={})", self.rank, self.size)
    }
}

impl Graph {
    fn describe(&self) {

    }
}

fn mainmodel(env: &EvolutionEnvironment) -> MainStochasticModel {
    let conf = MainModelConfig {
        n: 32,
        k: 4,

        n_mutations: 2,

        initial_probability_range: (0.0, 0.1),

        type_inhibitory_probability: 0.3,
        mutate_type_cpm_probability: 0.2,
        mutate_params_probability: 0.1,
        mutate_type_ratio_probability: 0.7
    };

    MainStochasticModel::new(env, &conf);
}



fn main() {
    init_logger(None);

    let env = EvolutionEnvironment {
        inputs: 1,
        outputs: 1,
    };

    let mut model = mainmodel(&env);
    let desc = model.sample();
    let graph: Graph = (&desc).into();

    println!("{}", graph);

    let edges = desc.edges();

    visualize_network(desc.n, edges);
}
