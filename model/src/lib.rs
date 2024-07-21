pub mod network;
pub mod neuron;
pub mod synapse;
pub mod spikes;
pub mod record;


use network::SpikingNetwork;
use neuron::izhikevich::Izhikevich;

use synapse::BaseSynapse;
use synapse::representation::MatrixRepresentation;

pub type DefaultNetwork = SpikingNetwork<Izhikevich, BaseSynapse<MatrixRepresentation>>;
