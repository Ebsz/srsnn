use evolution::Fitness;
use evolution::config::EvolutionConfig;

use tasks::TaskName;

use crate::evaluate::catching_evaluate;


pub struct RunConfig {
    pub task: TaskName,
    pub evolution_config: EvolutionConfig
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            task: TaskName::CatchingTask,
            evolution_config: EvolutionConfig::default()
        }
    }
}

//    fn (&mut self) -> dyn Task {
//        let task tasks::conf::get_task(self.taskname)
//    }
//
//    fn get_task_by_name(name: TaskName, conf: &dyn Any) -> Box<dyn Task> {
//        match name {
//            TaskName::CATCHING_TASK => {
//                let c: &CatchingTaskConfig = conf.downcast_ref().unwrap();
//                Box::new(CatchingTask::new(c))
//            }
//            TaskName::MOVEMENT_TASK => {
//                let c: MovementTaskConfig = conf.downcast_ref().unwrap();
//                Box::new(MovementTask::new(c))
//            }
//        }
//    }

//pub const DEFAULT_CONF = DefaultConfig<
//
//    let conf = Config {
//        task: TaskName::MOVEMENT_TASK
//    };
