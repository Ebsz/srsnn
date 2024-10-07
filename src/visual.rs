use crate::runnable::RunnableNetwork;

use tasks::Task;
use tasks::task_runner::TaskRunner;

use visual::tasks::TaskRenderer;
use visual::task_window::TaskWindow;

use model::network::representation::DefaultRepresentation;
use model::network::builder::NetworkBuilder;


pub fn visualize_network_on_task<T: Task + TaskRenderer>(task: T, repr: &DefaultRepresentation) {
    log::info!("Visualizing genome behavior on task");

    let network = NetworkBuilder::build(repr);
    let mut runnable = RunnableNetwork {
        network,
        inputs: repr.env.inputs,
        outputs: repr.env.outputs
    };

    let runner = TaskRunner::new(task, &mut runnable);

    let mut window = TaskWindow::new(runner);
    window.run();
}
