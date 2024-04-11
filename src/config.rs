use evolution::Fitness;

use tasks::TaskName;

//trait Config: Default {}
pub trait Config {}

pub struct RunConfig {
    //pub taskname: TaskName,
    pub fitness_fn: Fitness,
    pub taskname: TaskName
}

impl Config for RunConfig {}

//impl Default for RunConfig {
//    fn default() -> Self {
//        RunConfig {
//            //taskname: TaskName::CATCHING_TASK
//            fitness_fn
//        }
//    }
//}



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
