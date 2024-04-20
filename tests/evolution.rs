use evolution::EvolutionEnvironment;
use evolution::genome::Genome;
use evolution::population::Population;

use luna::config::RunConfig;


#[test]
fn evolve_small_population() {
    evolve_genome(2, 100);
}

#[test]
fn evolve_large_population() {
    evolve_genome(100, 100);
}

fn evolve_genome(n: usize, g: u32) {
    let mut config = RunConfig::default();
    config.evolution_config.population_size = n;
    config.evolution_config.max_generations = g;

    let env = EvolutionEnvironment {
        inputs: 1,
        outputs: 1,
        fitness: |_: &Genome, _: &EvolutionEnvironment| -> f32 { 0.0 }
    };

    let mut population = Population::new(env, config.evolution_config);

    let _genome = population.evolve();

    assert!((population.generation)  == g);
}
