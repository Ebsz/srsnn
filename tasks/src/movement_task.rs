use crate::cognitive_task::{CognitiveTask, TaskResult, TaskEnvironment, TaskInput, TaskState, TaskRenderer};

use ndarray::Array;

use sdl2::gfx::primitives::DrawRenderer;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;

use std::f32::consts::PI;

const WORLD_SIZE: (i16, i16) = (1000, 1000);

const AGENT_RADIUS: f32 = 32.0;
const AGENT_START_POS: (i16, i16) = (WORLD_SIZE.0 / 2, WORLD_SIZE.1 / 2);
const AGENT_START_ROTATION: f32 = PI; // / 2.0;
const AGENT_MOVEMENT_SPEED: f32 = 8.0;

const N_SENSORS: usize = 9;
const N_CONTROLS: usize = 4; // up/down + rotate left/right


pub struct MovementTask {
    agent: Agent,
    target: Target,
}

pub struct MovementTaskConfig { }


impl CognitiveTask for MovementTask {
    type TaskConfig = MovementTaskConfig;

    fn new(config: MovementTaskConfig) -> MovementTask {
        MovementTask {
            agent: Agent::new(),
            target: Target::new(100.0,100.0)
        }
    }
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState {

        self.parse_input(input);

        TaskState {
            result: None,
            sensor_data: Array::zeros(N_SENSORS)
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
                0 => {self.agent.rotation -= 0.15; },
                1 => {self.agent.rotation += 0.15; },
                2 => {self.agent.move_faced_direction(AGENT_MOVEMENT_SPEED); },
                3 => {self.agent.move_faced_direction(-AGENT_MOVEMENT_SPEED); },
                _ => {panic!("invalid input id"); }
            }
        }
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
        
        let _ = canvas.filled_circle(self.target.x as i16, self.target.y as i16, 50, Color::RED);

        const HAND_LEN: f32 = 300.0;

        let x = self.agent.x + self.agent.rotation.cos() * HAND_LEN;
        let y = self.agent.y - self.agent.rotation.sin() * HAND_LEN;

        let _ = canvas.thick_line(self.agent.x as i16, self.agent.y as i16, x as i16, y as i16, 3, Color::BLACK);
    }

    /// Returns the size of the 'arena' that the task operates in
    fn render_size() -> (i32, i32) {
        (WORLD_SIZE.0 as i32, WORLD_SIZE.1 as i32)
    }
}
