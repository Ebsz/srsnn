//! Evolutionary algorithm that evolves a spiking network.
//!
//! After initializing, the population is evolved by
//!     1. Calculating fitness
//!     2. Select best-fit genomes for reproduction
//!     3. Create new population from crossover
//!     4. Mutate new population
//!
//! This is repeated until for a set number of generations,
//! or until a genome is found that has a fitness > FITNESS_GOAL.
//!

pub mod genome;
pub mod config;

use genome::Genome;
use config::EvolutionConfig;

use utils::random;

use std::collections::HashMap;
use std::time::Instant;


#[derive(Debug, Clone)]
pub struct EvolutionEnvironment {
    pub inputs: usize,
    pub outputs: usize,
}

pub type Fitness = fn(&Genome, &EvolutionEnvironment) -> f32;

pub struct Population {
    population: HashMap<u32, Genome>,
    fitness_fn: Fitness,
    environment: EvolutionEnvironment,
    config: EvolutionConfig,
    generation: u32,
    genome_num: u32, // TODO: Find new name for this yo
}

impl Population {
    pub fn new(env: EvolutionEnvironment, f: Fitness, config: EvolutionConfig) -> Population {
        let mut population = HashMap::new();
        let mut genome_num: u32 = 0;

        log::debug!("Creating initial population");
        for _ in 0..config.population_size {
            population.insert(genome_num, Genome::new(&env, &config.genome_config));
            genome_num += 1;
        }

        Population {
            population,
            genome_num,
            environment: env,
            config,
            fitness_fn: f,
            generation: 0,
        }
    }

    pub fn evolve(&mut self) -> Genome {
        let mut fitness: Vec<(u32, f32)> = Vec::new();

        while self.generation < self.config.max_generations {
            log::debug!("Evaluating population");
            let start_time = Instant::now();

            // Evaluate the population and sort by fitness
            fitness = self.evaluate();
            fitness.sort_by(|x,y| y.1.partial_cmp(&x.1).unwrap());

            let eval_time = start_time.elapsed().as_secs_f32();
            let mean_fitness: f32 = fitness.iter().map(|(_, f)| f).sum::<f32>() / fitness.len() as f32;

            log::trace!("Evaluated population in {}s ({}s per genome)",
                eval_time, eval_time / self.config.population_size as f32 );
            log::info!("Generation {} - best fit: {}, mean: {}",
                self.generation, fitness[0].1, mean_fitness);

            if fitness[0].1 > self.config.fitness_goal {
                log::info!("Fitness goal reached, stopping evolution");
                break;
            }

            log::debug!("Creating new generation");

            // Select the best fit to be parents
            let n_parents = ((self.config.population_size as f32) * self.config.survival_threshold) as usize;
            let parents: Vec<u32> = fitness[0..n_parents].iter().map(|(i, _)| *i).collect();

            log::trace!("Breeding {} offspring from {} parents",
                (self.config.population_size - self.config.n_best_keep), parents.len());

            let mut children = self.breed(parents, self.config.population_size - self.config.n_best_keep);

            // Mutate children
            for c in &mut children {
                c.mutate(&self.config.genome_config);
            }

            // Remove all but the N best genomes from the population
            self.population.retain(|i, _| fitness[..self.config.n_best_keep].iter().any(|x| x.0 == *i));

            // Add new genomes to the population
            for g in children {
                self.add_genome(g);
            }

            assert!(self.population.len() == self.config.population_size);

            self.generation += 1;
        }

        // Return the best fit genome of the population
        self.population.get(&fitness[0].0).unwrap().clone()
    }

    /// Evaluate the fitness of each genome
    fn evaluate(&mut self) -> Vec<(u32, f32)> {
        let fitness: Vec<(u32, f32)> = self.population.iter()
            .map(|(id, g)| (*id, (self.fitness_fn)(g, &self.environment))).collect();

        fitness
    }

    /// Create n offspring from a set of parent genomes
    fn breed(&mut self, parent_ids: Vec<u32>, n: usize) -> Vec<Genome> {
        let mut offspring: Vec<Genome> = Vec::new();

        while offspring.len() < n {
            let id_1 = random::random_choice(&parent_ids);
            let id_2 = random::random_choice(&parent_ids);

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