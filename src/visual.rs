pub mod plots;

use crate::phenotype::EvolvableGenome;

use visual::window::TaskWindow;

use evolution::EvolutionEnvironment;

use tasks::{Task, TaskRenderer};
use tasks::task_runner::TaskRunner;


pub fn visualize_genome_on_task<G: EvolvableGenome, T: Task + TaskRenderer>(task: T, g: &G, env: &EvolutionEnvironment) {
    log::info!("Visualizing genome behavior on task");

    let mut phenotype = g.to_phenotype(env);

    let runner = TaskRunner::new(task, &mut phenotype);

    let mut window = TaskWindow::new(runner);
    window.run();
}
