use evolution::{EvolutionEnvironment, Fitness};
use evolution::genome::Genome;
use evolution::population::Population;

use luna::config::RunConfig;

use utils::random;


#[test]
fn population_can_evolve_and_return_a_genome() {
    let env = EvolutionEnvironment {
        inputs: 1,
        outputs: 1,
        fitness: |g: &Genome, e: &EvolutionEnvironment| -> f32 { 0.0 }
    };

    let mut config = RunConfig::default();
    config.evolution_config.population_size = 2;

    let mut population = Population::new(env, RunConfig::default().evolution_config);

    let genome = population.evolve();
}
