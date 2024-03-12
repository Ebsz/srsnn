
#[derive(Debug)]
pub struct TaskResult {
    pub success: bool,
    pub distance: f32,
}

pub trait CognitiveTask {
    fn tick(&mut self) -> Option<TaskResult>;

    //fn run(&mut self, phenotype: Phenotype);
}

pub struct TaskEnvironment {
    pub input: usize,
    pub output: usize,
}

