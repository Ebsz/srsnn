pub mod stochastic;
pub mod matrix;
pub mod srsnn;

use model::neuron::izhikevich::Izhikevich;
use model::network::representation::{NetworkRepresentation, NeuronDescription};

use utils::config::Configurable;


pub trait Model<N=NeuronDescription<Izhikevich>>: Configurable + Sync {
    fn develop(&self) -> NetworkRepresentation<N>;
}
