pub mod plots;

use crate::runnable::RunnableNetwork;

use model::neuron::NeuronModel;
use model::network::representation::{NetworkRepresentation, NeuronDescription};
use model::network::builder::NetworkBuilder;

use visual::fg::ForceGraphComponent;
use visual::window::{Window, WindowComponent};
use visual::task_window::TaskWindow;

use tasks::{Task, TaskRenderer};
use tasks::task_runner::TaskRunner;

pub fn visualize_network(n: usize, edges: Vec<(u32, u32)>) {
    let components: Vec<Box<dyn WindowComponent>> = vec![
        Box::new(ForceGraphComponent::new(n, edges))
    ];

    let mut w = Window::new(Window::DEFAULT_WINDOW_SIZE, components);

    w.run();
}

pub fn visualize_network_on_task<N: NeuronModel, T: Task + TaskRenderer>(task: T, desc: &NetworkRepresentation<NeuronDescription<N>>) {
    log::info!("Visualizing genome behavior on task");

    let network = NetworkBuilder::build(desc);
    let mut runnable = RunnableNetwork {
        network,
        inputs: desc.inputs,
        outputs: desc.outputs
    };

    let runner = TaskRunner::new(task, &mut runnable);

    let mut window = TaskWindow::new(runner);
    window.run();
}
