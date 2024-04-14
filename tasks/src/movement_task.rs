use crate::{Task, TaskResult, TaskEnvironment, TaskInput, TaskState, TaskRenderer};
use crate::sensor::Sensor;

use utils::random;

use ndarray::{Array, Array1};
use ndarray_rand::rand_distr::Uniform;

use sdl2::gfx::primitives::DrawRenderer;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;

use std::f32::consts::PI;


const WORLD_SIZE: (i16, i16) = (1000, 1000);

const AGENT_RADIUS: f32 = 32.0;
const AGENT_START_POS: (i16, i16) = (WORLD_SIZE.0 / 2, WORLD_SIZE.1 / 2);
const AGENT_START_ROTATION: f32 = PI; // / 2.0;
const AGENT_MOVEMENT_SPEED: f32 = 8.0;

const N_SENSORS: usize = 1;
const N_CONTROLS: usize = 4; // up/down + rotate left/right

const WALL_SIZE: usize = 50;

const SENSOR_LEN: f32 = 300.0;

const MAX_T: u32 = 300;
const TARGET_RADIUS: f32 = 50.0;


pub struct MovementTask {
    agent: Agent,
    target: Target,
    sensor: Sensor,
    ticks: u32,
    walls: Vec<Wall>
}

pub struct MovementTaskConfig { }


impl Task for MovementTask {
    type TaskConfig = MovementTaskConfig;

    fn new(config: MovementTaskConfig) -> MovementTask {

        MovementTask {
            agent: Agent::new(),
            target: Target::new(500.0,200.0),
            ticks: 0,
            sensor: Sensor::new(SENSOR_LEN, AGENT_START_ROTATION),

        }
    }
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState {
        if self.ticks >= MAX_T {
            return self.end_state();
        }

        self.parse_input(input);

        // TODO: Change random_matrix to accept shapes of any dim
        //let sensor_data: Array1<f32> = random::random_matrix((N_SENSORS, 1), Uniform::new(0.0, 0.5)).into_shape(N_SENSORS).unwrap() * 1.0;
        let sensor_data: Array1<f32> = Array::zeros(N_SENSORS);

        let sensor_read = self.read_sensors();


        self.ticks += 1;

        TaskState {
            result: None,
            sensor_data
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

    fn end_state(&self) -> TaskState {
        let distance = ((self.agent.x as f32 - AGENT_START_POS.0 as f32).powf(2.0)
            + (self.agent.y as f32 - AGENT_START_POS.1 as f32).powf(2.0)).sqrt();

        TaskState {
            result: Some(TaskResult {
                success: false,
                distance
            }),
            sensor_data: Array::zeros(N_SENSORS)
        }
    }

    fn read_sensors(&self) -> f32 {
        self.sensor.read((self.agent.x, self.agent.y), (self.target.x, self.target.y), TARGET_RADIUS)
    }
}

struct Agent {
    x: f32,
    y: f32,
    rotation: f32
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


enum TargetType {
    CIRCLE,
    SQUARE,
    TRIANGLE,
}

struct Target {
    x: f32,
    y: f32,
}

impl Target {
    fn new(x: f32, y: f32) -> Target {
        Target {
            x,
            y
        }
    }
}

impl TaskRenderer for MovementTask {
    fn render(&self, canvas: &mut WindowCanvas) {
        let _ = canvas.filled_circle(self.agent.x as i16, self.agent.y as i16, AGENT_RADIUS as i16, Color::BLACK);
        
        let _ = canvas.filled_circle(self.target.x as i16, self.target.y as i16, TARGET_RADIUS as i16, Color::RED);


        let x = self.agent.x + self.agent.rotation.cos() * SENSOR_LEN;
        let y = self.agent.y - self.agent.rotation.sin() * SENSOR_LEN;

        let _ = canvas.thick_line(self.agent.x as i16, self.agent.y as i16, x as i16, y as i16, 3, Color::BLACK);
    }

    /// Returns the size of the 'arena' that the task operates in
    fn render_size() -> (i32, i32) {
        (WORLD_SIZE.0 as i32, WORLD_SIZE.1 as i32)
    }
}
