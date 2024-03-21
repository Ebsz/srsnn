use ndarray::Array1;

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

pub struct TaskContext {
    pub agent_inputs: usize,
    pub agent_outputs: usize,
}

pub trait CognitiveTask {
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState;

    fn context() -> TaskContext;
}
