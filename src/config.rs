use evolution::config::EvolutionConfig;

use tasks::TaskName;


pub struct RunConfig {
    pub task: TaskName,
    pub evolution_config: EvolutionConfig,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            task: TaskName::SurvivalTask,
            evolution_config: EvolutionConfig::default()
        }
    }
}
