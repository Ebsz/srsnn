//! A cognitive task where the agent is to catch a falling apple
//! NOTE: The coordinate system used here assumes y = 0 is at the top in accordance with
//! how it's used in graphics libraries like macroquad.
//! TODO: Change the coordinate system to correspond with human perception

use crate::cognitive_task::{CognitiveTask, TaskResult, TaskContext, TaskInput, TaskState};

use ndarray::{array, Array, Array1};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};

use std::f32::consts::PI;

const ARENA_SIZE: (i32, i32) = (800, 600);
const APPLE_RADIUS: i32 = 16;
const AGENT_RADIUS: i32 = 32;

// How many units to move per tick
const APPLE_SPEED: i32 = 3;
const AGENT_SPEED: i32 = 5;

const AGENT_START_POS: (i32, i32) = (ARENA_SIZE.0 / 2, (ARENA_SIZE.1 - AGENT_RADIUS));

const N_SENSORS: usize = 7;
const SENSOR_SPREAD: f32 = PI / 2.0; // The angle between the first and last sensor

const N_AGENT_CONTROLS: usize = 2;

// TODO: This is not respected by the algorithm for some reason. Investigate
pub const SENSOR_LEN: f32 = 1000.0;

pub struct CatchingTask {
    pub agent: Agent,
    pub apple: Apple,
    pub sensors: Vec<Sensor>,
    ticks: usize,
}

impl CognitiveTask for CatchingTask {
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState {
        self.parse_input(input);

        self.apple.y += APPLE_SPEED;

        let mut result: Option<TaskResult> = None;

        if self.intersects() || self.apple.y  > ARENA_SIZE.1 {
            result =  Some(TaskResult {
                success: self.intersects(),
                distance: self.distance()
            });
        }

        let sensor_data = self.read_sensors();

        self.ticks += 1;

        TaskState {
            result,
            sensor_data,

        }
    }

    fn context() -> TaskContext {
        TaskContext {
            agent_inputs: N_SENSORS,
            agent_outputs: N_AGENT_CONTROLS,
        }
    }
}

impl CatchingTask {
    pub fn new() -> CatchingTask {
        let agent = Agent::new();
        let apple = Apple::new();

        CatchingTask {
            ticks: 0,
            sensors: CatchingTask::init_sensors(),
            agent,
            apple
        }
    }

    pub fn read_sensors(&self) -> Array1<f32> {
        let mut readout = Array::zeros(N_SENSORS);

        let pos = (self.agent.x as f32, self.agent.y as f32);
        let target_pos = (self.apple.x as f32, self.apple.y as f32);

        for (i, s) in self.sensors.iter().enumerate() {
            readout[i] = s.read(pos, target_pos);
        }

        readout
    }

    fn init_sensors() -> Vec<Sensor> {
        let mut sensors = Vec::new();

        for i in 0..N_SENSORS {
            let alpha: f32 = (PI - SENSOR_SPREAD) / 2.0;
            let angle = alpha + (SENSOR_SPREAD/(N_SENSORS-1) as f32) * i as f32;

            sensors.push(Sensor::new(angle));
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
    fn new() -> Apple {
        let mut rng = StdRng::seed_from_u64(0);

        let x: i32 = rng.gen_range(0..ARENA_SIZE.0);
        let y: i32 = 0;

        Apple {
            r: APPLE_RADIUS,
            x,
            y,
        }
    }
}

// Sensors are rays out from the center of the agent
pub struct Sensor {
    pub angle: f32,
}


impl Sensor {
    pub fn new(angle: f32) -> Sensor {
        Sensor {
            angle
        }
    }

    pub fn endpoint(&self, agent_pos: (f32,f32)) -> (f32, f32) {
        let x = agent_pos.0 + self.angle.cos() * SENSOR_LEN;
        let y = agent_pos.1 - self.angle.sin() * SENSOR_LEN;

        (x, y)
    }

    pub fn read(&self, pos: (f32, f32), target_pos: (f32, f32)) -> f32 {
        // TODO: Currently does not respect sensor length.
        // Subtract the target pos to center the target at (0,0), which simplifies the calculation
        let p = array![pos.0 - target_pos.0, pos.1 - target_pos.1];


        let q = array![pos.0 + self.angle.cos() * SENSOR_LEN - target_pos.0,
                 pos.1 + self.angle.sin() * SENSOR_LEN - target_pos.1];

        //println!("({:?}, {:?}), ({:?}, {:?})", p[0], p[1], q[0], q[1]);
        //let len = ((p[0] - q[0]).powf(2.0) + (p[1] - q[1]).powf(2.0)).sqrt();
        //println!("{:?}", len);

        let d = &q - &p;

        let a = &d.dot(&d);
        let b = 2.0 * &p.dot(&d);
        let c = &p.dot(&p) - (APPLE_RADIUS as f32).powf(2.0);

        let det = b * b - 4.0 * a * c;

        if det > 0.0 {
            return ((pos.0 - target_pos.0).powf(2.0) + (pos.1 - target_pos.1).powf(2.0)).sqrt();

            //println!("{:?}", a);
            //let lambda = (-b + (b*b - 4.0 * a * c).sqrt())/ 2.0 * a;
            //return lambda;
        }

        0.0
    }
}
