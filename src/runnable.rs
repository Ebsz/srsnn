//! Wraps a Network to enable running on tasks

use model::network::Network;
use model::spikes::Spikes;

use tasks::{TaskInput, TaskOutput};
use tasks::task_runner::Runnable;

use model::DefaultNetwork;
use model::network::representation::DefaultRepresentation;
use model::network::builder::NetworkBuilder;

use ndarray::Array1;


pub struct RunnableNetwork<N: Network> {
    pub network: N,
    pub inputs: usize,
    pub outputs: usize
}

impl<N: Network> Runnable for RunnableNetwork<N> {
    fn step(&mut self, output: TaskOutput) -> Vec<u32> {
        let network_input = self.task_output_to_network_input(output);

        let network_state = self.network.step(network_input); // network_state: len(N)

        self.get_output(network_state)
    }

    fn reset(&mut self) {
        self.network.reset_state();
    }
}


impl<N: Network> RunnableNetwork<N> {
    pub fn build(repr: &DefaultRepresentation) -> RunnableNetwork<DefaultNetwork> {
        let network = NetworkBuilder::build(repr);

        RunnableNetwork {
            network,
            inputs: repr.env.inputs,
            outputs: repr.env.outputs,
        }
    }

    fn task_output_to_network_input(&self, output: TaskOutput) -> Spikes {
        // Ensure that the sensor input is boolean
        let task_data: Array1<bool> = output.data.mapv(|x| if x != 0.0 { true } else { false });

        assert!(task_data.shape()[0] == self.inputs,
            "expected network input of size {}, got {}", self.inputs, task_data.shape()[0]);

        let mut network_input = Spikes::new(self.inputs);
        network_input.data.assign(&task_data);

        network_input
    }

    fn get_output(&self, network_state: Spikes) -> Vec<u32> {
        (0..self.outputs).zip(network_state.data)
            .filter_map(|(i, f)| if f {Some(i as u32)} else { None })
            .collect()

        //(&network_state).into()
    }

    //fn network_state_to_task_input(&self, network_state: Spikes) -> Vec<TaskInput> {
    //    let mut task_inputs: Vec<TaskInput> = Vec::new();

    //    //log::info!("{}",network_state.data.len()); = n  = (128)

    //    // Parse the firing state of output neurons to commands
    //    for i in 0..self.outputs as usize {
    //        // If the neurons assigned to be output fire, add the corresponding input
    //        if network_state.data[i] {
    //            task_inputs.push(TaskInput { input_id: i as u32});
    //        }
    //    }

    //    task_inputs
    //}
}
