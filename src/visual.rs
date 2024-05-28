pub mod plots;

use crate::phenotype::EvolvableGenome;

use model::network::description::NetworkDescription;

use visual::fg::ForceGraphComponent;
use visual::window::{Window, WindowComponent};
use visual::task_window::TaskWindow;

use evolution::EvolutionEnvironment;

use tasks::{Task, TaskRenderer};
use tasks::task_runner::TaskRunner;

pub fn visualize_network<N>(desc: NetworkDescription<N>) {
    let edges = desc.edges();

    let components: Vec<Box<dyn WindowComponent>> = vec![
        Box::new(ForceGraphComponent::new(desc.n, edges))
    ];

    let mut w = Window::new(Window::DEFAULT_WINDOW_SIZE, components);

    println!("Running..");
    w.run();
}

pub fn visualize_genome_on_task<G: EvolvableGenome, T: Task + TaskRenderer>(task: T, g: &G, env: &EvolutionEnvironment) {
    log::info!("Visualizing genome behavior on task");

    let mut phenotype = g.to_phenotype(env);

    let runner = TaskRunner::new(task, &mut phenotype);

    let mut window = TaskWindow::new(runner);
    window.run();
}
