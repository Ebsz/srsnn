use crate::{Task, TaskEval, TaskInput, TaskOutput, TaskState, TaskEnvironment};

use utils::{math, encoding};

use ndarray::{s, Axis, Array, Array1, Array2};


const N_CLASSES: usize = 10;
const OUTPUTS_PER_CLASS: usize = 2;

const SEND_TIME: u32 = 50;        // How long each pattern is sent.
const SEND_DELAY: u32 = 70;       // Delay between patterns
const RESPONSE_WINDOW: u32 = 50;

const RESPONSE_START_T: u32 = SEND_TIME + SEND_DELAY;
const MAX_T: u32 = RESPONSE_START_T + RESPONSE_WINDOW;

const AGENT_INPUTS: usize = 784;
const AGENT_OUTPUTS: usize = N_CLASSES * OUTPUTS_PER_CLASS;

const MAX_FIRING_PROBABILITY: f32 = 0.5;


#[derive(Debug)]
pub struct MNISTResult {
    response: Array2<u32>, // T X Spikes
    label: Array1<f32>
}

#[derive(Clone)]
pub struct MNISTSetup {
    pub pattern: Array1<f32>, // 28 x 28
    pub label: Array1<f32>, // One-hot encoded label
}

pub struct MNISTTask {
    setup: MNISTSetup,
    t: u32,

    response: Array2<u32>
}

impl Task for MNISTTask {
    type Setup = MNISTSetup;
    type Result = MNISTResult;

    fn new(setup: &Self::Setup) -> Self {


        MNISTTask {
            setup: setup.clone(), // TODO: This clone might slow things down considerably
            t: 0,
            response: Array::zeros((RESPONSE_WINDOW as usize, AGENT_OUTPUTS))
        }
    }

    fn tick(&mut self, input: TaskInput) -> TaskState<Self::Result> {
        if self.t >= MAX_T {
            return TaskState {
                output: self.get_output(),
                result: Some(MNISTResult { response: self.response.clone(), label: self.setup.label.to_owned() })
            }
        }

        if self.t >= MAX_T - RESPONSE_WINDOW {
            self.save_response(&input.data);
        }

        self.t += 1;

        TaskState {
            output: self.get_output(),
            result: None
        }
    }

    fn reset(&mut self) {

    }

    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: AGENT_INPUTS,
            agent_outputs: AGENT_OUTPUTS,
        }
    }
}

impl MNISTTask {
    fn save_response(&mut self, input: &[u32]) {
        let t = (self.t - RESPONSE_START_T) as usize;

        for i in input {
            self.response[[t, *i as usize]] = 1;
        }
    }

    fn get_output(&mut self) -> TaskOutput {
        let data = if self.t < SEND_TIME {
            encoding::rate_encode(&(&self.setup.pattern * MAX_FIRING_PROBABILITY))
        } else {
            Array::zeros(AGENT_INPUTS)
        };

        //if self.t < INPUT_SEND_TIME {
        //    let ix = self.t * AGENT_INPUTS;
        //    //let data = self.setup.input.slice(s![ix..ix+AGENT_INPUTS]).to_owned();
        //    return TaskOutput { data };
        //}
        //log::info!("{}", data.len());

        TaskOutput { data }
    }
}

impl TaskEval for MNISTTask {
    fn eval_setups() -> Vec<Self::Setup> {
        log::info!("Loading MNIST data");
        let (data, labels) = mnist::load_mnist().expect("Could not load MNIST data");

        //log::trace!("Rate-encoding MNIST data");
        //let encoded_data = encoding::rate_encode_array(&(&data * 0.8));
        //assert!(encoded_data.shape() == data.shape());

        let mut setups = Vec::new();
        for i in 0..mnist::N_IMAGES_TRAIN {
            setups.push( MNISTSetup {
                pattern: data.slice(s![i, ..]).to_owned(),
                label: labels.slice(s![i, ..]).to_owned(),
            });
        }

        setups
    }

    fn fitness(results: Vec<Self::Result>) -> f32 {
        let mut fitness = 0.0;
        let batch_size = results.len();

        //let mut correct = 0;

        for r in &results {
            // Number of spikes per output neuron
            let sum = r.response.sum_axis(Axis(0));

            // Number of spikes per label
            let label_sum: Array1<u32> = sum.exact_chunks(OUTPUTS_PER_CLASS)
                .into_iter().map(|x| x.sum()).collect();

            assert!(label_sum.shape()[0] == N_CLASSES);

            // Divide by the total time to get firing rates
            let firing_rates: Array1<f32> = label_sum.map(|x| *x as f32)
                / (OUTPUTS_PER_CLASS as f32 * r.response.shape()[0] as f32);

            let predictions = math::ml::softmax(&firing_rates);

            let error = math::ml::cross_entropy(&predictions, &r.label);

            //log::warn!("{}", predictions);

            let predicted_label = math::max_index(&firing_rates);
            let label = math::max_index(&r.label);

            let all_labels_equal = firing_rates.windows(2).into_iter().all(|x| x[0] == x[1]);
            if predicted_label == label && !all_labels_equal {
                //correct += 1;

                fitness += 5.0;
            }

            fitness += 5.0 - math::minf(&[error, 5.0]);
        }

        fitness = fitness / (10.0 * batch_size as f32) * 100.0;

        //log::debug!("{}/{} correct", correct, results.len());

        fitness
    }

    fn accuracy(_results: &[Self::Result]) -> Option<f32> {
        None
    }
}

//fn softmax(x: &Array1<f32>) -> Array1<f32> {
//    let mut s = x.mapv(f32::exp);
//
//    s /= s.sum();
//
//    s
//}
//
//fn cross_entropy(predictions: &Array1<f32>, label: &Array1<f32>) -> f32 {
//    -(label * &predictions.mapv(f32::ln).view()).sum()
//}

mod mnist {
    use std::fs::File;
    use std::io::Read;

    use ndarray::{Array, Array2};

    pub const IMG_SIZE: usize = 784;

    pub const LABEL_HEADER_LEN: usize = 8;
    pub const IMG_HEADER_LEN: usize = 16;
    pub const N_IMAGES_TRAIN: usize = 60000;

    pub fn load_mnist() -> Result<(Array2<f32>, Array2<f32>), String> {
        let training_data: (Vec<u8>, Vec<u8>) = load_training_set()?;

        let img: Array2<f32> = normalize_img(data_to_arrays(training_data.0));
        let labels: Array2<f32> = one_hot_encode(training_data.1);

        assert!(img.shape() == [N_IMAGES_TRAIN, IMG_SIZE]);

        Ok((img, labels))
    }

    fn one_hot_encode(labels: Vec<u8>) -> Array2<f32> {
        let mut onehot: Array2<f32> = Array2::zeros((labels.len(), 10));

        for (n, l) in labels.into_iter().enumerate() {
            onehot.row_mut(n)[l as usize] = 1.0;
        }

        onehot
    }

    /// Normalize each point in the images from [0,255] to (0,1)
    fn normalize_img(data: Array2<f32>) -> Array2<f32> {
        data / 255.0
    }

    /// Convert the raw image data to arrays that can be batched
    /// Returns an Array2 of shape [60000, 784]
    fn data_to_arrays(data: Vec<u8>) -> Array2<f32> {
        Array::from_shape_vec((60000, IMG_SIZE), data)
            .unwrap()
            .mapv(|x| f32::from(x))
    }

    fn load_training_set() -> Result<(Vec<u8>, Vec<u8>), String> {
        let mut img = read_file("mnist/train-images-idx3-ubyte")?;
        let mut labels = read_file("mnist/train-labels-idx1-ubyte")?;

        // Remove file headers
        img.drain(..IMG_HEADER_LEN);
        labels.drain(..LABEL_HEADER_LEN);

        assert_eq!(img.len(), N_IMAGES_TRAIN * IMG_SIZE);
        assert_eq!(labels.len(), N_IMAGES_TRAIN);

        Ok((img, labels))
    }

    fn read_file(path: &'static str) -> Result<Vec<u8>, String> {
        let file = File::open(path);

        match file {
            Ok(mut f) => {
                let mut buffer: Vec<u8> = Vec::new();
                let _ = f.read_to_end(&mut buffer);

                Ok(buffer)
            }
            Err(e) => {
                Err(format!("Could not read file {path}: {e}"))
            }
        }
    }
}
