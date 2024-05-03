pub mod sensor;
pub mod task_runner;

pub mod catching_task;
pub mod movement_task;
pub mod survival_task;

use ndarray::Array1;
use sdl2::render::WindowCanvas;


pub struct TaskInput {
    pub input_id: i32
}

pub struct TaskEnvironment {
    pub agent_inputs: usize,
    pub agent_outputs: usize,
}

#[derive(Debug)]
pub struct TaskState<R> {
    pub result: Option<R>,
    pub sensor_data: Array1<f32>
}

pub trait Task {
    type Setup;
    type Result;

    fn new(setup: &Self::Setup) -> Self;
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState<Self::Result>;
    fn reset(&mut self);
    fn environment() -> TaskEnvironment;
}


pub trait TaskEval: Task {
    fn eval_setups() -> Vec<Self::Setup>;

    fn fitness(results: Vec<Self::Result>) -> f32;
}

/// The TaskRenderer trait is implemented by a Task in order
/// to define its visual representation.
pub trait TaskRenderer {
    fn render(&self, canvas: &mut WindowCanvas);

    /// Returns the size of the 'arena' that the task operates in
    fn render_size() -> (i32, i32);
}
