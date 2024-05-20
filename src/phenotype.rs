use model::network::Network;
use model::spikes::Spikes;

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;

use tasks::{TaskInput, TaskOutput};
use tasks::task_runner::Runnable;

use ndarray::Array1;


pub trait EvolvableGenome: Genome + Sized {
    type Phenotype: Runnable;

    fn to_phenotype(&self, env: &EvolutionEnvironment) -> Self::Phenotype;
}

pub struct Phenotype<N: Network> {
    network: N,

    network_inputs: usize,
    network_outputs: usize,
}

impl<N: Network> Phenotype<N> {
    pub fn new(network: N, env: EvolutionEnvironment) -> Phenotype<N> {
        Phenotype {
            network,

            network_inputs: env.inputs,
            network_outputs: env.outputs,
        }
    }

    fn task_output_to_network_input(&self, output: TaskOutput) -> Spikes {
        // Ensure that the sensor input is boolean
        let task_data: Array1<bool> = output.data.mapv(|x| if x != 0.0 { true } else { false });

        let mut network_input = Spikes::new(self.network_inputs);
        network_input.data.assign(&task_data);

        network_input
    }

    fn network_state_to_task_input(&self, network_state: Spikes) -> Vec<TaskInput> {
        let mut task_inputs: Vec<TaskInput> = Vec::new();

        // Parse the firing state of output neurons to commands
        for i in 0..self.network_outputs {
            // TODO: unnecessary cast; make phenotype.outputs be usize
            // If the neurons assigned to be output fire, add the corresponding input
            if network_state.data[i as usize] {
                task_inputs.push(TaskInput { input_id: i as i32});
            }
        }

        task_inputs
    }
}

impl<N: Network> Runnable for Phenotype<N> {
    fn step(&mut self, output: TaskOutput) -> Vec<TaskInput> {
        let network_input = self.task_output_to_network_input(output);

        let network_state = self.network.step(network_input); // network_state: len(N)

        self.network_state_to_task_input(network_state)
    }

    fn reset(&mut self) {
        //TODO: Add reset fn to network trait and implement
    }
}
