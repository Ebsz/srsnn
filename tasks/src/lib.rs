pub mod catching_task;
pub mod movement_task;
pub mod sensor;

use ndarray::Array1;
use sdl2::render::WindowCanvas;

use catching_task::CatchingTask;
use movement_task::MovementTask;

#[derive(Clone, Copy, Debug)]
pub enum TaskName {
    CatchingTask,
    MovementTask,
}

pub fn get_environment(task: TaskName) -> TaskEnvironment {
    match task {
        TaskName::CatchingTask => CatchingTask::environment(),
        TaskName::MovementTask => MovementTask::environment()
    }
}

#[derive(Debug)]
pub struct TaskState {
    pub result: Option<TaskResult>,
    pub sensor_data: Array1<f32>
}

#[derive(Debug)]
pub struct TaskResult {
    pub success: bool,
    pub distance: f32,
}

pub struct TaskInput {
    pub input_id: i32
}

pub struct TaskEnvironment {
    pub agent_inputs: usize,
    pub agent_outputs: usize,
}

pub trait Task {
    type TaskConfig; // TODO: rename to TaskSetup to differentiate from configs

    fn new(config: Self::TaskConfig) -> Self;
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState;
    fn reset(&mut self);
    fn environment() -> TaskEnvironment;
}

/// The TaskRenderer trait is implemented by a Task in order
/// to define its visual representation.
pub trait TaskRenderer {
    fn render(&self, canvas: &mut WindowCanvas);

    /// Returns the size of the 'arena' that the task operates in
    fn render_size() -> (i32, i32);
}
