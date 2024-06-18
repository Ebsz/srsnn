use crate::{Task, TaskEnvironment, TaskInput, TaskOutput, TaskState, TaskEval};
use crate::sensor::Sensor;

use ndarray::{Array, Array1};

use std::f32::consts::PI;

pub const AGENT_RADIUS: f32 = 32.0;
pub const TARGET_RADIUS: f32 = 50.0;
pub const SENSOR_LEN: f32 = 300.0;
pub const WORLD_SIZE: (i16, i16) = (1000, 1000);

const AGENT_START_POS: (i16, i16) = (WORLD_SIZE.0 / 2, WORLD_SIZE.1 / 2);
const AGENT_START_ROTATION: f32 = PI; // / 2.0;
const AGENT_MOVEMENT_SPEED: f32 = 8.0;

const N_SENSORS: usize = 1;
const N_CONTROLS: usize = 4; // up/down + rotate left/right


const MAX_T: u32 = 300;


pub struct MovementTask {
    pub agent: Agent,
    pub target: Target,
    pub sensor: Sensor,
    pub ticks: u32,
}

#[derive(Clone)]
pub struct MovementTaskSetup { }

#[derive(Debug)]
pub struct MovementTaskResult {
    pub distance: f32,
}

impl Task for MovementTask {
    type Setup = MovementTaskSetup;
    type Result = MovementTaskResult;

    fn new(_setup: &MovementTaskSetup) -> MovementTask {

        MovementTask {
            agent: Agent::new(),
            target: Target::new(500.0,200.0),
            ticks: 0,
            sensor: Sensor::new(SENSOR_LEN, AGENT_START_ROTATION),

        }
    }
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState<MovementTaskResult> {
        if self.ticks >= MAX_T {
            return self.end_state();
        }

        self.parse_input(input);

        // TODO: Change random_matrix to accept shapes of any dim
        //let sensor_data: Array1<f32> = random::random_matrix((N_SENSORS, 1), Uniform::new(0.0, 0.5)).into_shape(N_SENSORS).unwrap() * 1.0;
        let sensor_data: Array1<f32> = self.read_sensors();

        self.ticks += 1;

        TaskState {
            result: None,
            output: TaskOutput { data: sensor_data }
        }
    }
    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: N_SENSORS,
            agent_outputs: N_CONTROLS,
        }
    }

    fn reset(&mut self) {

    }
}

impl MovementTask {
    fn parse_input(&mut self, input: &Vec<TaskInput>) {
        for i in input {
            match i.input_id {
                0 => {self.agent.rotation -= 0.10; self.sensor.angle = self.agent.rotation; },
                1 => {self.agent.rotation += 0.10; self.sensor.angle = self.agent.rotation; },
                2 => {self.agent.move_faced_direction(AGENT_MOVEMENT_SPEED); },
                3 => {self.agent.move_faced_direction(-AGENT_MOVEMENT_SPEED); },
                _ => {panic!("invalid input id"); }
            }
        }
    }

    fn end_state(&self) -> TaskState<MovementTaskResult> {
        let distance = ((self.agent.x as f32 - AGENT_START_POS.0 as f32).powf(2.0)
            + (self.agent.y as f32 - AGENT_START_POS.1 as f32).powf(2.0)).sqrt();

        TaskState {
            result: Some(MovementTaskResult {
                distance
            }),
            output: TaskOutput { data: Array::zeros(N_SENSORS) }
        }
    }

    fn read_sensors(&self) -> Array1<f32> {
        let mut sensor_data: Array1<f32> = Array::zeros(N_SENSORS);

        let d = self.sensor.read((self.agent.x, self.agent.y), (self.target.x, self.target.y), TARGET_RADIUS);
        sensor_data[0] = d;

        sensor_data
    }
}

pub struct Agent {
    pub x: f32,
    pub y: f32,
    pub rotation: f32
}

impl Agent {
    fn new() -> Agent {
        Agent {
            x: AGENT_START_POS.0 as f32,
            y: AGENT_START_POS.1 as f32,
            rotation: AGENT_START_ROTATION,
        }
    }

    // Move the agent forward/backward in the direction faced
    fn move_faced_direction(&mut self, amount: f32) {
        let dx = self.rotation.cos() * amount;
        let dy = self.rotation.sin() * amount;


        let new_x = self.x + dx;
        let new_y = self.y - dy;

        // TODO: This is imprecise, allowing the agent only to approach the wall but not touch it
        //       Dividing radius by 2 permits stepping into the wall - this avoid that, at least
        if new_x > (WORLD_SIZE.0 as f32) - AGENT_RADIUS
            || new_x - AGENT_RADIUS < 0.0
            || new_y > (WORLD_SIZE.1 as f32) - AGENT_RADIUS
            || new_y - AGENT_RADIUS < 0.0
        {
            return;
        }
        
        self.x += dx;
        self.y -= dy;
    }
}

pub struct Target {
    pub x: f32,
    pub y: f32,
}

impl Target {
    fn new(x: f32, y: f32) -> Target {
        Target {
            x,
            y
        }
    }
}

impl TaskEval for MovementTask {
    fn eval_setups() -> Vec<MovementTaskSetup> {
        vec![MovementTaskSetup {}]
    }

    fn fitness(_results: Vec<MovementTaskResult>) -> f32 {
        0.0
    }
}
