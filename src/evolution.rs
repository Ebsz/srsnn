//! Evolutionary algorithm that evolves a spiking network on some task
//!
//! After initializing the population, the algorithm performs the following steps
//!     1. Calculate fitness
//!     2. Create new population from crossover
//!     3. Mutate

// TODO: Implement some kind of trait to contain the fitness function
//       Inspired by darwin-rs: Individual {mutate(), calculate_fitness(), reset()}

pub mod genome;
pub mod phenotype;

use genome::Genome;

use std::collections::HashMap;

// Evolution parameters
const POPULATION_SIZE: usize = 10;
const N_PARENTS_KEEP: usize = 20;

#[derive(Debug, Clone)]
pub struct EvolutionEnvironment {
    pub inputs: usize,
    pub outputs: usize,
}


pub type Fitness = fn(&Genome) -> f32;

pub struct Population {
    population: HashMap<u32, Genome>,
    fitness_fn: Fitness,
    environment: EvolutionEnvironment,
    generation: u32,
    genome_num: u32, // TODO: Find new name for this yo
}

impl Population {
    pub fn new(env: EvolutionEnvironment, f: Fitness) -> Population {

        let mut genome_num: u32 = 0;

        // Build initial population
        let mut population = HashMap::new();

        // TODO: Ensure diversity by creating all different genomes to begin with
        for i in 0..POPULATION_SIZE {
            population.insert(genome_num, Genome::new(&env));
            genome_num += 1;
        }

        Population {
            population,
            genome_num,
            environment: env,
            fitness_fn: f,
            generation: 0,
        }
    }

    pub fn evolve(&mut self) {
        //let fitness = (self.fitness_fn)(&self.population[&(0 as u32)]);


        let fitness: Vec<(u32, f32)> = self.population.iter()
            .map(|(id, g)| (*id, (self.fitness_fn)(g))).collect();


        println!("{:?}", fitness);

        // Selection: sort by fitness and select N best to keep
    }
}
