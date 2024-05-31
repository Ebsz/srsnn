pub mod stochastic;
pub mod matrix;

use model::neuron::izhikevich::Izhikevich;
use model::network::representation::{NetworkRepresentation, NeuronDescription};

use evolution::genome::Genome;


pub trait Model<N=NeuronDescription<Izhikevich>>: Genome {
    fn develop(&self) -> NetworkRepresentation<N>;
}
