use tasks::{Task, TaskEval};

#[derive(Clone)]
pub enum EvalSetup<T: Task + TaskEval> {
    Base(Vec<T::Setup>),
    Batched(BatchSetup<T>),
}

impl<T: Task + TaskEval> EvalSetup<T> {
    pub fn get(&self) -> &[T::Setup] {
        match self {
            EvalSetup::Base(setup) => &setup,
            EvalSetup::Batched(b) => {
                &b.setups[b.batch_index..(b.batch_index + b.batch_size)]
            }
        }
    }

    pub fn next(&mut self) {
        if let EvalSetup::Batched(b) = self {
            b.batch_index = (b.batch_index + b.batch_size) % b.setups.len();
        }
    }
}

#[derive(Clone)]
pub struct BatchSetup<T: Task + TaskEval> {
    setups: Vec<T::Setup>,
    batch_size: usize,
    batch_index: usize
}

impl<T: Task + TaskEval> BatchSetup<T> {
    pub fn new(setups: Vec<T::Setup>, batch_size: usize) -> BatchSetup<T> {
        BatchSetup {
            setups,
            batch_size,
            batch_index: 0
        }
    }
}
