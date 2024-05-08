use crate::genome::Genome;

use crate::EvolutionEnvironment;

use utils::config::ConfigSection;

use serde::Deserialize;



struct PoolGenome {

}

#[derive(Deserialize)]
struct PoolGenomeConfig {
    n_pools: usize
}

impl ConfigSection for PoolGenomeConfig {
    fn name() -> String {
        "pool".to_string()
    }
}


//impl<P: Pool> Genome for PoolGenome {
//    type Config  = PoolGenomeConfig;
//
//    fn new(env: &EvolutionEnvironment, config: &Self::Config) -> Self {
//        PoolGenome {
//
//        }
//    }
//
//    fn mutate(&mut self, config: &Self::Config) {
//
//    }
//
//    fn crossover(&self, other: &PoolGenome) -> Self {
//        PoolGenome {
//
//        }
//    }
//}

//pub trait Gene {
//    fn mutate(&mut self);
//
//    fn crossover(&mut self, other: Self);
//}
//
//
//trait Pool {
//    fn step(&mut self, input: f32) {
//
//    }
//}
//
//impl<P: Pool> Gene for P {
//    fn mutate(mut self) {
//    }
//
//    fn crossover(&mut self, other: Self) {
//    }
//}
//
//
//trait BasePool {
//
//}



//impl EvolvableGenome for PoolGenome {
//    type Phenotype = PoolPhenotype;
//
//    fn to_phenotype(&self) -> Self::Phenotype {
//
//    }
//}
//
//struct PoolPhenotype;
//
//impl Runnable for PoolPhenotype {
//    fn step(&mut self, task_output: Array1<f32>) -> Vec<TaskInput> {
//
//    }
//}
//
//
//struct SubNetwork {
//
//}
//
//
//impl SubNetwork {
//
//}

//impl<N: NeuronModel, S: Synapse> Runnable for Network<N, S> {
//
//}

//struct NetworkConnection {
//    from: (u32, u32),
//    to: (u32, u32),
//    w: f32
//}
//
//
//struct MultiNetwork<N: NeuronModel, S: Synapse>  {
//    subnetworks: Vec<Network<N, S>>,
//    input_network: u32,
//    output_network: u32,
//
//    network_connections: Vec<NetworkConnection>
//}
//
//
//impl<N: NeuronModel, S: Synapse> MultiNetwork<N, S> {
//
//    fn step(&mut self, input: Array1<f32>
//}
