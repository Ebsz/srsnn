use crate::runnable::RunnableNetwork;

use tasks::Task;
use tasks::task_runner::TaskRunner;

use visual::tasks::TaskRenderer;
use visual::task_window::TaskWindow;

use model::network::SpikingNetwork;
use model::network::representation::DefaultRepresentation;
use model::network::builder::NetworkBuilder;
use model::neuron::izhikevich::Izhikevich;
use model::synapse::Synapse;


pub fn visualize_network_on_task<T: Task + TaskRenderer, S: Synapse>(task: T, repr: &DefaultRepresentation) {
    log::info!("Visualizing genome behavior on task");

    let network: SpikingNetwork<Izhikevich, S> = NetworkBuilder::build(repr);
    let mut runnable = RunnableNetwork {
        network,
        inputs: repr.env.inputs,
        outputs: repr.env.outputs
    };

    let runner = TaskRunner::new(task, &mut runnable);

    let mut window = TaskWindow::new(runner);
    window.run();
}
