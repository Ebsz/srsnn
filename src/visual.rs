pub mod window;
pub mod plots;

use crate::phenotype::Phenotype;
use crate::visual::window::TaskWindow;

use evolution::EvolutionEnvironment;
use evolution::genome::Genome;

use tasks::{Task, TaskRenderer, TaskResult};
use tasks::task_runner::TaskRunner;

pub fn visualize_genome_on_task<T, R: TaskResult>(task: T, g: &Genome, env: &EvolutionEnvironment)
where
    T: Task<R> + TaskRenderer

{
    log::info!("Visualizing genome behavior on task");

    let mut phenotype = Phenotype::from_genome(g, env);
    let runner = TaskRunner::new(task, &mut phenotype);

    let mut window = TaskWindow::new(runner);
    window.run();
}
