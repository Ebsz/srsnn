pub mod window;
pub mod plots;

use crate::phenotype::Phenotype;
use crate::task_executor::TaskExecutor;
use crate::visual::window::TaskWindow;

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;

use tasks::{Task, TaskRenderer};

pub fn visualize_genome_on_task<T: Task + TaskRenderer>(task: T, g: &Genome, env: &EvolutionEnvironment) {
    log::info!("Visualizing genome behavior on task");

    let mut phenotype = Phenotype::from_genome(g, env);
    let executor = TaskExecutor::new(task, &mut phenotype);

    let mut window = TaskWindow::new(executor);
    window.run();
}
