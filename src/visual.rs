use crate::runnable::RunnableNetwork;

use tasks::Task;
use tasks::task_runner::TaskRunner;

use visual::tasks::TaskRenderer;
use visual::task_window::TaskWindow;

use model::neuron::NeuronModel;
use model::network::representation::{NetworkRepresentation, NeuronDescription};
use model::network::builder::NetworkBuilder;


pub fn visualize_network_on_task<N: NeuronModel, T: Task + TaskRenderer>(task: T, repr: &NetworkRepresentation<NeuronDescription<N>>) {
    log::info!("Visualizing genome behavior on task");

    let network = NetworkBuilder::build(repr);
    let mut runnable = RunnableNetwork {
        network,
        inputs: repr.inputs,
        outputs: repr.outputs
    };

    let runner = TaskRunner::new(task, &mut runnable);

    let mut window = TaskWindow::new(runner);
    window.run();
}
