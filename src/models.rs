pub mod stochastic;
pub mod matrix;

use model::neuron::NeuronModel;
use model::neuron::izhikevich::Izhikevich;
use model::network::description::{NetworkDescription, NeuronDescription};

use evolution::genome::{Genome, EvolvableGenome};


pub trait Model<N=NeuronDescription<Izhikevich>>: Genome {
    fn develop(&self) -> NetworkDescription<N>;
}

pub struct EvolvableModel<M, N> {
    model: M,
    n: N
}

impl<M: Model<N>, N: NeuronModel> EvolvableGenome for EvolvableModel<M, N> {
    type Phenotype = NetworkDescription<N>;

    fn to_phenotype(&self)-> NetworkDescription<N> {
        self.model.develop()
    }
}
