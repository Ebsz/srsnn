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
use tasks::catching_task::{CatchingTask, CatchingTaskConfig};

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
fn evaluate(g: &Genome, env: &EvolutionEnvironment) -> f32 {
    const trial_positions: [i32; 11] = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];

    let mut phenotype = Phenotype::from_genome(g, env);

    let max_distance = tasks::catching_task::ARENA_SIZE.0 as f32;

    let mut total_fitness = 0.0;
    for i in 0..trial_positions.len() {
        phenotype.reset();

        let task_conf = CatchingTaskConfig {
            target_pos: trial_positions[i]
        };

        let task = CatchingTask::new(task_conf);

        let result = execute(&mut phenotype, task);
        total_fitness += (1.0 - result.distance/max_distance) * 100.0 - (if result.success {0.0} else {30.0});
    }

    let fitness = total_fitness / trial_positions.len() as f32;

    log::trace!("eval: {:?}", fitness);

    fitness
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
