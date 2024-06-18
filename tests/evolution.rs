use evolution::{Evaluate, EvolutionEnvironment};
use luna::models::matrix::MatrixModel;
use evolution::population::Population;

use luna::config::{main_config, get_config, MainConfig};
use utils::config::Configurable;


fn get_test_config(n: usize, g: u32) -> (MainConfig, <MatrixModel as Configurable>::Config) {
    let mut main_config = main_config(None);

    main_config.evolution.population_size = n;
    main_config.evolution.max_generations = g;

    let mut genome_config = get_config::<MatrixModel>();

    genome_config.max_neurons  = 10;
    genome_config.initial_neuron_count_range = (2, 4);
    genome_config.initial_connection_count_range = (5, 10);

    (main_config, genome_config)
}

#[test]
fn evolve_small_population() {
    evolve_genome(2, 100);
}

#[test]
fn evolve_large_population() {
    evolve_genome(100, 100);
}

fn evolve_genome(n: usize, g: u32) {
    let config = get_test_config(n, g);
    let main_config = config.0;
    let genome_config = config.1;

    let env = EvolutionEnvironment {
        inputs: 1,
        outputs: 1,
    };

    struct TestEvaluator;
    impl Evaluate<MatrixModel, ()> for TestEvaluator {
        fn eval(&mut self, m: &[(u32, &MatrixModel)]) -> Vec<(u32, f32, ())> {
            m.iter().map(|x| (x.0, 0.0, ())).collect()
        }
    }

    let mut population = Population::new(env, main_config.evolution, genome_config, TestEvaluator {});

    let _genome = population.evolve();

    assert!((population.generation)  == g);
}
