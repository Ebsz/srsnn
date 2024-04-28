use evolution::EvolutionEnvironment;
use evolution::genome::Genome;
use evolution::population::Population;

use luna::config::{get_config, MainConfig};


fn get_test_config(n: usize, g: u32) -> MainConfig {
    let mut config = get_config(None);

    config.evolution.population_size = n;
    config.evolution.max_generations = g;

    config.genome.max_neurons  = 10;
    config.genome.initial_neuron_count_range = (2, 4);
    config.genome.initial_connection_count_range = (5, 10);

    config
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

    let env = EvolutionEnvironment {
        inputs: 1,
        outputs: 1,
        fitness: |_: &Genome, _: &EvolutionEnvironment| -> f32 { 0.0 }
    };

    let mut population = Population::new(env, config.evolution, config.genome);

    let _genome = population.evolve();

    assert!((population.generation)  == g);
}
