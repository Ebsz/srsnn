use luna::pools::IzhikevichPool;
use luna::network::Network;
use luna::plots::generate_plots;
use luna::record::Record;

use luna::evolution::{Population, EvolutionEnvironment, Fitness};
use luna::evolution::genome::Genome;
use luna::evolution::phenotype::Phenotype;

use luna::task_executor::execute;

use luna::logger::init_logger;

use tasks::cognitive_task::{CognitiveTask, TaskResult};
use tasks::catching_task::CatchingTask;

use std::time::Instant;
use ndarray::{Array, Array2};


#[allow(dead_code)]
fn run() {
    const N: usize = 100; // # of neuron
    const T: usize = 300; // # of steps to run for
    const P: f32 = 0.1;   // The probability that two arbitrary neurons are connected
                          //
    let input: Array2<f32> = Array::ones((T, N)) * 17.3;

    let mut pool = IzhikevichPool::linear_pool(N, P);
    //let mut pool = IzhikevichPool::matrix_pool(N);
    let mut record = Record::new();

    println!("Running..");

    let start_time = Instant::now();
    pool.run(T, input, &mut record);

    println!("Simulated {} neurons for {} steps in {}s", N, T, (start_time.elapsed().as_secs_f32()));

    generate_plots(&record);
}

#[allow(dead_code)]
fn evaluate(g: &Genome) -> f32 {
    let phenotype = Phenotype::from_genome(g);

    let task = CatchingTask::new();

    let result = execute(phenotype, task);
    println!("result: {:?}", result);

    0.0
}

#[allow(dead_code)]
fn evolve() {
    let task_context = CatchingTask::context();

    let env = EvolutionEnvironment {
        inputs: task_context.agent_inputs,
        outputs: task_context.agent_outputs
    };

    let mut population = Population::new(env, evaluate);

    log::info!("Evolving population..");
    population.evolve();
}

fn main() {
    init_logger();

    run();
    //evolve();
}
