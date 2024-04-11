pub mod catching_task;
pub mod movement_task;

use ndarray::Array1;
use sdl2::render::WindowCanvas;

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
    type TaskConfig;

    fn new(config: Self::TaskConfig) -> Self;
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState;
    fn environment() -> TaskEnvironment;
    fn reset(&mut self);
}

/// The TaskRenderer trait is implemented by a Task in order
/// to define its visual representation.
pub trait TaskRenderer {
    fn render(&self, canvas: &mut WindowCanvas);

    /// Returns the size of the 'arena' that the task operates in
    fn render_size() -> (i32, i32);
}
