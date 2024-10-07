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
                let setups = &b.setups[b.batch_index..(b.batch_index + b.batch_size)];

                assert!(setups.len() == b.batch_size);

                setups
            }
        }
    }

    pub fn next(&mut self) {
        if let EvalSetup::Batched(b) = self {
            b.batch_index = (b.batch_index + b.batch_size) % b.setups.len();
        }
    }

    pub fn validation_setups(&self) -> &[T::Setup] {
        match self {
            EvalSetup::Base(_) => { &[] },
            EvalSetup::Batched(b) => { &b.validation_setups },
        }
    }
}

#[derive(Clone)]
pub struct BatchSetup<T: Task + TaskEval> {
    setups: Vec<T::Setup>,
    validation_setups: Vec<T::Setup>,
    batch_size: usize,
    batch_index: usize
}

impl<T: Task + TaskEval> BatchSetup<T> {
    pub fn new(setups: Vec<T::Setup>, batch_size: usize) -> BatchSetup<T> {
        // Split into validation and training setups
        //const VALIDATION_FRACTION: f32 = 0.20;
        //let n_train_batches = ((setups.len() as f32 / batch_size as f32)
        //* (1.0 - VALIDATION_FRACTION)) as usize;
        //let p = n_train_batches * batch_size;

        let train = setups.clone(); //setups[..p].to_vec();
        let val = setups; //[p..].to_vec();

        log::debug!("splitting dataset - n training: {}, n validation: {}", train.len(), val.len());

        assert!(train.len() % batch_size == 0,
            "train.len() % batch_size !=0 - train len: {}, batch_size: {}", train.len(), batch_size);

        BatchSetup {
            setups: train,
            validation_setups: val,

            batch_size,
            batch_index: 0
        }
    }
}
