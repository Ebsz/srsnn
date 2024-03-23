//! Evolutionary algorithm that evolves a spiking network on some task
//!
//! After initializing the population, the algorithm performs the following steps
//!     1. Calculate fitness
//!     2. Create new population from crossover
//!     3. Mutate

pub mod genome;
pub mod phenotype;

use std::collections::HashMap;
use genome::Genome;
use crate::utils;


const POPULATION_SIZE: usize = 50;
const MAX_GENERATIONS: u32 = 50;
const N_BEST_KEEP: usize = 0; // N best genomes to keep without mutation in the new population
const SURVIVAL_THRESHOLD: f32 = 0.3;


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
        let mut population = HashMap::new();
        let mut genome_num: u32 = 0;

        // Build initial population
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
        while self.generation < MAX_GENERATIONS {
            log::debug!("Evaluating population");

            // Evaluate the population and sort by fitness
            let mut fitness: Vec<(u32, f32)> = self.evaluate();
            fitness.sort_by(|x,y| y.1.partial_cmp(&x.1).unwrap());

            log::info!("Generation {} - best: {}", self.generation, fitness[0].1);

            // Select the best fit to be parents
            let n_parents = ((POPULATION_SIZE as f32) * SURVIVAL_THRESHOLD) as usize;
            let parents: Vec<u32> = fitness[0..n_parents].iter().map(|(i, _)| *i).collect();

            log::trace!("Breeding {} offspring from {} parents",
                (POPULATION_SIZE - N_BEST_KEEP), parents.len());

            let children = self.breed(parents, (POPULATION_SIZE - N_BEST_KEEP));

            // Remove all but the N best genomes from the population
            self.population.retain(|i, g| fitness[..N_BEST_KEEP].iter().any(|x| x.0 == *i));

            // Add new genomes to the population
            for g in children {
                self.add_genome(g);
            }

            assert!(self.population.len() == POPULATION_SIZE);

            self.generation += 1;
        }
    }

    /// Evaluate the fitness of each genome
    fn evaluate(&mut self) -> Vec<(u32, f32)> {
        let mut fitness: Vec<(u32, f32)> = self.population.iter()
            .map(|(id, g)| (*id, (self.fitness_fn)(g))).collect();

        fitness
    }


    /// Create n offspring from a set of parent genomes
    fn breed(&mut self, parent_ids: Vec<u32>, n: usize) -> Vec<Genome> {
        let mut offspring: Vec<Genome> = Vec::new();

        while offspring.len() < n {
            let id_1 = utils::random_choice(&parent_ids);
            let id_2 = utils::random_choice(&parent_ids);

            if id_1 == id_2 {
                continue;
            }

            let g1 = self.population.get(id_1).unwrap();
            let g2 = self.population.get(id_2).unwrap();

            offspring.push(g1.crossover(g2));
        }

        assert!(offspring.len() == n);

        offspring
    }

    fn add_genome(&mut self, g: Genome) {
        self.population.insert(self.genome_num, g);

        self.genome_num +=1;
    }

}
