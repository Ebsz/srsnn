pub mod sensor;
pub mod task_runner;

pub mod catching_task;
pub mod movement_task;
pub mod survival_task;

use catching_task::CatchingTask;
use movement_task::MovementTask;
use survival_task::SurvivalTask;

use ndarray::Array1;
use sdl2::render::WindowCanvas;


#[derive(Clone, Copy, Debug)]
pub enum TaskName {
    CatchingTask,
    MovementTask,
    SurvivalTask
}

pub struct TaskInput {
    pub input_id: i32
}

pub struct TaskEnvironment {
    pub agent_inputs: usize,
    pub agent_outputs: usize,
}

pub trait TaskResult {}

#[derive(Debug)]
pub struct TaskState<R: TaskResult> {
    pub result: Option<R>,
    pub sensor_data: Array1<f32>
}

pub trait Task<R: TaskResult> {
    type TaskConfig;

    fn new(config: Self::TaskConfig) -> Self;
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState<R>;
    fn reset(&mut self);
    fn environment() -> TaskEnvironment;
}

pub fn get_environment(task: TaskName) -> TaskEnvironment {
    match task {
        TaskName::CatchingTask => CatchingTask::environment(),
        TaskName::MovementTask => MovementTask::environment(),
        TaskName::SurvivalTask => SurvivalTask::environment()
    }
}

/// The TaskRenderer trait is implemented by a Task in order
/// to define its visual representation.
pub trait TaskRenderer {
    fn render(&self, canvas: &mut WindowCanvas);

    /// Returns the size of the 'arena' that the task operates in
    fn render_size() -> (i32, i32);
}
