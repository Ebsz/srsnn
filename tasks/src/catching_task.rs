//! A cognitive task where the agent is to catch a falling apple
//! NOTE: The coordinate system used here assumes y = 0 is at the top in accordance with
//! how it's used in graphics libraries like macroquad.
//! TODO: Change the coordinate system to correspond with human perception

use crate::{Task, TaskEnvironment, TaskInput, TaskOutput, TaskState, TaskEval};
use crate::sensor::Sensor;

use ndarray::{Array, Array1};

use std::f32::consts::PI;

pub const ARENA_SIZE: (i32, i32) = (500, 600);
pub const APPLE_RADIUS: i32 = 16;
pub const AGENT_RADIUS: i32 = 32;

const AGENT_START_POS: (i32, i32) = (ARENA_SIZE.0 / 2, (ARENA_SIZE.1 - AGENT_RADIUS));
const N_AGENT_CONTROLS: usize = 2;
const APPLE_SPEED: i32 = 3;
const AGENT_SPEED: i32 = 5;

const N_SENSORS: usize = 7;
const SENSOR_SPREAD: f32 = PI / 3.0; // The angle between the first and last sensor

const SENSOR_LEN: f32 = 800.0;


pub struct CatchingTask {
    pub agent: Agent,
    pub apple: Apple,
    pub sensors: Vec<Sensor>,
    pub setup: CatchingTaskSetup,
    ticks: usize,
}

#[derive(Copy, Clone)]
pub struct CatchingTaskSetup {
    pub target_pos: i32
}

#[derive(Debug)]
pub struct CatchingTaskResult {
    pub success: bool,
    pub distance: f32,
}

impl Task for CatchingTask {
    type Setup = CatchingTaskSetup;
    type Result = CatchingTaskResult;

    fn new(setup: &CatchingTaskSetup) -> CatchingTask {
        assert!(setup.target_pos <= ARENA_SIZE.0 && setup.target_pos >= 0);

        let agent = Agent::new();
        let apple = Apple::new(setup.target_pos);

        CatchingTask {
            ticks: 0,
            sensors: CatchingTask::init_sensors(),
            agent,
            setup: *setup,
            apple
        }
    }

    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState<CatchingTaskResult> {
        self.parse_input(input);

        self.apple.y += APPLE_SPEED;

        let mut result: Option<CatchingTaskResult> = None;

        if self.intersects() || self.apple.y  > ARENA_SIZE.1 {
            result =  Some(CatchingTaskResult {
                success: self.intersects(),
                distance: self.distance()
            });
        }

        let sensor_data = self.read_sensors();

        self.ticks += 1;

        TaskState {
            result,
            output: TaskOutput { data: sensor_data }
        }
    }

    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: N_SENSORS,
            agent_outputs: N_AGENT_CONTROLS,
        }
    }

    fn reset(&mut self) {
        self.agent = Agent::new();
        self.apple = Apple::new(self.setup.target_pos);
        self.ticks = 0;
    }
}

impl CatchingTask {
    pub fn read_sensors(&self) -> Array1<f32> {
        let mut readout = Array::zeros(N_SENSORS);

        let pos = (self.agent.x as f32, self.agent.y as f32);
        let target_pos = (self.apple.x as f32, self.apple.y as f32);

        for (i, s) in self.sensors.iter().enumerate() {
            readout[i] = s.read(pos, target_pos, APPLE_RADIUS as f32);
        }

        readout
    }

    fn init_sensors() -> Vec<Sensor> {
        let mut sensors = Vec::new();

        for i in 0..N_SENSORS {
            let alpha: f32 = (PI - SENSOR_SPREAD) / 2.0;
            let angle = alpha + (SENSOR_SPREAD/(N_SENSORS-1) as f32) * i as f32;

            sensors.push(Sensor::new(SENSOR_LEN, angle));
        }

        sensors
    }

    fn parse_input(&mut self, input: &Vec<TaskInput>) {
        for i in input {
            if i.input_id == 0 {
                self.agent.move_right();
            } else if i.input_id == 1 {
                self.agent.move_left();
            } else {
                panic!("Got input id > 1")
            }
        }
    }

    /// Euclidean distance between agent and apple
    fn distance(&self) -> f32 {
        (((self.agent.x - self.apple.x) as f32).powf(2.0) +
         ((self.agent.y - self.apple.y) as f32).powf(2.0)).sqrt()
    }

    /// Check if agent and apple intersects
    fn intersects(&self) -> bool {
        self.distance() <= (self.agent.r + self.apple.r) as f32
    }
}

pub struct Agent {
    pub r: i32,
    pub x: i32,
    pub y: i32,
}

pub struct Apple {
    pub r: i32,
    pub x: i32,
    pub y: i32,
}

impl Agent {
    pub fn new() -> Agent {
        Agent {
            r: AGENT_RADIUS,
            x: AGENT_START_POS.0,
            y: AGENT_START_POS.1
        }
    }

    pub fn move_left(&mut self) {
        if (self.x - AGENT_SPEED) < 0 {
            return;
        }

        self.x -= AGENT_SPEED;
    }
    pub fn move_right(&mut self) {
        if (self.x + AGENT_SPEED) > ARENA_SIZE.0 {
            return;
        }

        self.x += AGENT_SPEED;
    }
}


impl Apple {
    fn new(x_pos: i32) -> Apple {

        Apple {
            r: APPLE_RADIUS,
            x: x_pos,
            y: 0,
        }
    }
}

impl TaskEval for CatchingTask {
    fn eval_setups() -> Vec<CatchingTaskSetup> {
        let trial_positions: [i32; 11] = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];

        let mut setups = vec![];

        for p in trial_positions {
            setups.push(CatchingTaskSetup {
                target_pos: p
            });
        }

        setups
    }

    fn fitness(results: Vec<CatchingTaskResult>) -> f32 {
        let max_distance = ARENA_SIZE.0 as f32;

        let mut total_fitness: f32 = 0.0;
        let mut correct = 0;

        for r in &results {
            total_fitness += (1.0 - r.distance/max_distance) * 100.0 - (if r.success {0.0} else {30.0});

            if r.success {
                correct += 1;
            }
        }

        //log::debug!("{}/{} correct", correct, results.len());

        total_fitness / results.len() as f32
    }
}
