//! A cognitive task where the agent is to catch a falling apple
//! NOTE: The coordinate system used here assumes y = 0 is at the top in accordance with
//! how it's used in graphics libraries like macroquad.
//! TODO: Change the coordinate system to correspond with human perception

use crate::{Task, TaskResult, TaskEnvironment, TaskInput, TaskState, TaskRenderer};

use crate::sensor::Sensor;

use sdl2::render::WindowCanvas;
use sdl2::gfx::primitives::DrawRenderer;
use sdl2::pixels::Color;

use ndarray::{array, Array, Array1};

use std::f32::consts::PI;

const ARENA_SIZE: (i32, i32) = (500, 600);

const AGENT_START_POS: (i32, i32) = (ARENA_SIZE.0 / 2, (ARENA_SIZE.1 - AGENT_RADIUS));
const AGENT_RADIUS: i32 = 32;
const APPLE_SPEED: i32 = 3;
const N_AGENT_CONTROLS: usize = 2;

const APPLE_RADIUS: i32 = 16;
const AGENT_SPEED: i32 = 5;


const N_SENSORS: usize = 7;
const SENSOR_SPREAD: f32 = PI / 5.0; // The angle between the first and last sensor


const SENSOR_LEN: f32 = 600.0;

pub struct CatchingTask {
    pub agent: Agent,
    pub apple: Apple,
    pub sensors: Vec<Sensor>,
    pub config: CatchingTaskConfig,
    ticks: usize,
}

pub struct CatchingTaskConfig {
    pub target_pos: i32
}

impl Task for CatchingTask {
    type TaskConfig = CatchingTaskConfig;

    fn new(config: CatchingTaskConfig) -> CatchingTask {
        assert!(config.target_pos <= ARENA_SIZE.0 && config.target_pos >= 0);

        let agent = Agent::new();
        let apple = Apple::new(config.target_pos);

        CatchingTask {
            ticks: 0,
            sensors: CatchingTask::init_sensors(),
            agent,
            config,
            apple
        }
    }

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

    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: N_SENSORS,
            agent_outputs: N_AGENT_CONTROLS,
        }
    }

    fn reset(&mut self) {
        self.agent = Agent::new();
        self.apple = Apple::new(self.config.target_pos);
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


impl TaskRenderer for CatchingTask {
    fn render(&self, canvas: &mut WindowCanvas) {
        let _ = canvas.filled_circle(self.apple.x as i16, self.apple.y as i16, APPLE_RADIUS as i16, Color::RED);
        let _ = canvas.filled_circle(self.agent.x as i16, self.agent.y as i16, AGENT_RADIUS as i16, Color::BLACK);


        // Draw Sensors
        let agent_pos = (self.agent.x as f32, self.agent.y as f32);

        for s in &self.sensors {
            let sensor_endpoint = s.endpoint(agent_pos);

            let _ = canvas.thick_line(agent_pos.0 as i16, agent_pos.1 as i16,
                sensor_endpoint.0 as i16, sensor_endpoint.1 as i16, 2, Color::BLACK);
        }
    }

    fn render_size() -> (i32, i32) {
        return ARENA_SIZE;
    }
}
