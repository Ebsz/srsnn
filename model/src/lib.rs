pub mod network;
pub mod neuron;
pub mod synapse;
pub mod spikes;
pub mod record;

use network::SpikingNetwork;
use network::representation::{NetworkRepresentation, NeuronDescription};

use neuron::izhikevich::Izhikevich;

use synapse::BaseSynapse;
use synapse::representation::MatrixRepresentation;

use utils::config::Configurable;
use utils::parameters::ParameterSet;
use utils::environment::Environment;


pub type DefaultNetwork = SpikingNetwork<Izhikevich, BaseSynapse<MatrixRepresentation>>;
pub type DefaultNeuron = NeuronDescription<Izhikevich>;

pub trait Model<N=DefaultNeuron>: Configurable + Sync {
    fn new(config: &Self::Config, p: &ParameterSet, env: &Environment) -> Self;

    fn develop(&self) -> NetworkRepresentation<N>;

    fn params(config: &Self::Config) -> ParameterSet;
}

