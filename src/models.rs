pub mod stochastic;
pub mod matrix;

use model::neuron::NeuronModel;
use model::neuron::izhikevich::Izhikevich;
use model::network::description::NetworkDescription;

use evolution::genome::EvolvableGenome;


pub trait Model<N=Izhikevich> {
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
