use crate::genome::Genome;

use crate::{Evaluate, EvolutionEnvironment};
use crate::config::EvolutionConfig;
use crate::stats::EvolutionStatistics;

use utils::random;

use std::time::Instant;
use std::cmp::max;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;


const STALE_THRESHOLD: f32 = 0.1;

pub struct Population<E: Evaluate<G, P>, G: Genome, P> {
    pub generation: u32,
    pub stop_signal: Arc<AtomicBool>,
    pub stats: EvolutionStatistics,

    population: Vec<Individual<G, P>>,

    environment: EvolutionEnvironment,
    config: EvolutionConfig,

    genome_config: G::Config,
    genome_num: u32, // TODO: Find new name for this yo
                     //
    evaluator: E, // TODO: Figure out if this should just be injected instead.
}

impl<E: Evaluate<G, P>, G: Genome, P> Population<E, G, P> {
    pub fn new(env: EvolutionEnvironment, config: EvolutionConfig, genome_config: G::Config, evaluator: E) -> Population<E, G, P> {
        assert!(env.inputs > 0 && env.outputs > 0);
        assert!(config.population_size >= 2);

        let mut population = Vec::new();
        let mut genome_num: u32 = 0;

        log::debug!("Creating initial population");
        for _ in 0..config.population_size {
            population.push( Individual::new(G::new(&env, &genome_config), genome_num));

            genome_num += 1;
        }

        Population {
            population,
            genome_num,
            environment: env,
            generation: 0,
            stop_signal: Arc::new(AtomicBool::new(false)),

            config,
            genome_config,

            evaluator,
            stats: EvolutionStatistics::new()
        }
    }

    pub fn evolve(&mut self) -> &Individual<G, P> {
        let mut sorted_fitness: Vec<(u32, f32)>;

        let mut stale_counter = 0;
        let mut last_gen_best_fitness: f32 = 0.0;

        loop {
            self.evaluate();
            sorted_fitness = self.get_sorted_fitness();

            if sorted_fitness[0].1 - last_gen_best_fitness < STALE_THRESHOLD {
                stale_counter += 1
            } else {
                stale_counter = 0;
            }

            if self.config.n_stale_before_reset > 0 && stale_counter >= self.config.n_stale_before_reset {
                log::debug!("Population is stale, resetting");
                self.reset_population(&sorted_fitness);

                stale_counter = 0;
                continue;
            }

            self.log_generation(&sorted_fitness);

            if self.should_stop(&sorted_fitness) {
                break;
            }

            log::debug!("Creating new generation");
            let new_generation = self.reproduce(&sorted_fitness);

            self.population.retain(|g| sorted_fitness[..self.config.elites].iter().any(|x| x.0 == g.id));

            // Add new genomes to the population
            for g in new_generation {
                self.add_genome(g);
            }

            assert!(self.population.len() == self.config.population_size);

            last_gen_best_fitness = sorted_fitness[0].1;

            self.generation += 1;
        }

        // Return the best fit genome of the population
        &self.population.iter().find(|g| g.id == sorted_fitness[0].0).unwrap()
    }

    /// Reset the population by generating a new population, keeping only the elites
    fn reset_population(&mut self, sorted_fitness: &Vec<(u32, f32)>) {
        self.population.retain(|g| sorted_fitness[..self.config.elites].iter().any(|x| x.0 == g.id));

        for _ in 0..(self.config.population_size-self.config.elites) {
            self.population.push( Individual::new(G::new(&self.environment, &self.genome_config), self.genome_num));

            self.genome_num += 1;
        }
    }

    /// Evaluate the fitness of each genome
    fn evaluate(&mut self) {
        log::debug!("Evaluating population");

        let start_time = Instant::now();

        let genomes = self.population
            .iter()
            .filter_map(|i| if  i.fitness == None { Some((i.id, &i.genome)) } else { None } )
            .collect::<Vec<(u32, &G)>>();
        let evals = self.evaluator.eval(&genomes[..]);

        // Update population with the new evals
        for eval in evals {
            let mut g = self.population.iter_mut().find(|g| g.id == eval.0).unwrap();
            g.fitness = Some(eval.1);
            g.phenotype = Some(eval.2);
        }

        let eval_time = start_time.elapsed().as_secs_f32();

        log::debug!("Evaluated population in {}s", eval_time);
    }

    fn get_sorted_fitness(&self) -> Vec<(u32, f32)> {
        let mut sorted_fitness: Vec<(u32, f32)> = self.population.iter()
            .map(|g| (g.id, g.fitness.unwrap())).collect();

        sorted_fitness.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

        assert!(sorted_fitness.windows(2).all(|f| f[0].1 >= f[1].1));

        sorted_fitness
    }

    fn reproduce(&mut self, fitness: &Vec<(u32, f32)>) -> Vec<G> {
        let n_parents = max(2, ((self.config.population_size as f32) * self.config.parent_fraction) as usize);

        assert!(n_parents >= 2);

        let parent_ids: Vec<u32> = fitness[0..n_parents].iter().map(|(i, _)| *i).collect();

        log::trace!("Breeding {} offspring from {} parents",
            (self.config.population_size - self.config.elites), parent_ids.len());

        let mut children = self.breed(parent_ids, self.config.population_size - self.config.elites);

        // Mutate children
        for c in &mut children {
            c.mutate(&self.genome_config);
        }

        children
    }

    /// Create n offspring from a set of parent genomes
    fn breed(&mut self, parent_ids: Vec<u32>, n: usize) -> Vec<G> {
        assert!(parent_ids.len() != 0);

        let mut offspring: Vec<G> = Vec::new();

        while offspring.len() < n {
            let id_1 = random::random_choice(&parent_ids);
            let id_2 = random::random_choice(&parent_ids);

            if id_1 == id_2 {
                continue;
            }

            let g1 = self.population.iter().find(|g| g.id == *id_1).unwrap();
            let g2 = self.population.iter().find(|g| g.id == *id_2).unwrap();

            let g: G;

            if g1.fitness > g2.fitness {
                g = g1.genome.crossover(&g2.genome);
            } else {
                g = g2.genome.crossover(&g1.genome);
            }

            offspring.push(g);
        }

        assert!(offspring.len() == n);

        offspring
    }

    fn add_genome(&mut self, g: G) {
        self.population.push( Individual::new(g, self.genome_num) );

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

    fn log_generation(&mut self, sorted_fitness: &Vec<(u32, f32)>) {
        let mean_fitness: f32 = sorted_fitness.iter().map(|(_, f)| f).sum::<f32>() / sorted_fitness.len() as f32;
        let best_fitness: f32 = sorted_fitness[0].1;

        log::info!("Generation {} - best fit: {}, mean: {}",
            self.generation, best_fitness, mean_fitness);

        self.stats.log_generation(best_fitness, mean_fitness);
    }
}

/// Represents a member of the population
pub struct Individual<G: Genome, P> {
    pub genome: G,
    pub id: u32,
    pub fitness: Option<f32>,
    pub phenotype: Option<P>
}

impl<G: Genome, P> Individual<G, P> {
    fn new(genome: G, id: u32) -> Individual<G, P> {
        Individual {
            genome,
            id,
            fitness: None,
            phenotype: None
        }
    }
}
