pub mod stochastic;
pub mod matrix;
pub mod srsnn;

use model::neuron::izhikevich::Izhikevich;
use model::network::representation::{NetworkRepresentation, NeuronDescription};

use evolution::genome::Genome;


pub trait Model<N=NeuronDescription<Izhikevich>>: Genome + Sync{
    fn develop(&self) -> NetworkRepresentation<N>;
}
