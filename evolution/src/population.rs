use crate::genome::Genome;

use crate::EvolutionEnvironment;
use crate::config::{EvolutionConfig, GenomeConfig};

use utils::random;

use std::collections::HashMap;
use std::time::Instant;
use std::cmp::max;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct Population {
    pub generation: u32,
    pub stop_signal: Arc<AtomicBool>,

    population: HashMap<u32, Genome>,
    environment: EvolutionEnvironment,
    config: EvolutionConfig,

    genome_config: GenomeConfig,
    genome_num: u32, // TODO: Find new name for this yo
}

impl Population {
    pub fn new(env: EvolutionEnvironment, config: EvolutionConfig, genome_config: GenomeConfig) -> Population {
        assert!(env.inputs > 0 && env.outputs > 0);
        assert!(config.population_size >= 2);

        let mut population = HashMap::new();
        let mut genome_num: u32 = 0;

        log::debug!("Creating initial population");
        for _ in 0..config.population_size {
            population.insert(genome_num, Genome::new(&env, &genome_config));
            genome_num += 1;
        }

        Population {
            population,
            genome_num,
            environment: env,
            generation: 0,
            stop_signal: Arc::new(AtomicBool::new(false)),

            config,
            genome_config
        }
    }

    pub fn evolve(&mut self) -> Genome {
        let mut fitness: Vec<(u32, f32)>;

        loop {
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

            if self.should_stop(&fitness) {
                break;
            }

            log::debug!("Creating new generation");
            let new_generation = self.reproduce(&fitness);

            self.population.retain(|i, _| fitness[..self.config.elites].iter().any(|x| x.0 == *i));

            // Add new genomes to the population
            for g in new_generation {
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
            .map(|(id, g)| (*id, (self.environment.fitness)(g, &self.environment))).collect();

        fitness
    }

    fn reproduce(&mut self, fitness: &Vec<(u32, f32)>) -> Vec<Genome> {
        // Select the best fit to be parents
        let n_parents = max(2, ((self.config.population_size as f32) * self.config.parent_fraction) as usize);

        assert!(n_parents >= 2);

        let parents: Vec<u32> = fitness[0..n_parents].iter().map(|(i, _)| *i).collect();

        log::trace!("Breeding {} offspring from {} parents",
            (self.config.population_size - self.config.elites), parents.len());

        let mut children = self.breed(parents, self.config.population_size - self.config.elites);

        // Mutate children
        for c in &mut children {
            c.mutate(&self.genome_config);
        }

        children
    }

    /// Create n offspring from a set of parent genomes
    fn breed(&mut self, parent_ids: Vec<u32>, n: usize) -> Vec<Genome> {
        assert!(parent_ids.len() != 0);

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

    fn should_stop(&self, sorted_fitness: &Vec<(u32, f32)>) -> bool {
        if sorted_fitness[0].1 > self.config.fitness_goal {
            log::info!("Fitness goal reached, stopping evolution");
            return true;
        }

        if self.generation >= self.config.max_generations {
            log::info!("Max generations reached");
            return true;
        }

        if self.stop_signal.load(Ordering::SeqCst) {
            log::trace!("Stop signal received");
            return true;
        }

        false
    }
}
