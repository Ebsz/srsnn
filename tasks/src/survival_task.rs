use crate::{Task, TaskEnvironment, TaskInput, TaskOutput, TaskState, TaskEval};
use crate::sensor::Sensor;

use utils::random;

use ndarray::{Array, Array1};

use std::f32::consts::PI;


pub const WORLD_SIZE: (i16, i16) = (1000, 1000);
pub const SENSOR_LEN: f32 = 500.0;

const N_CONTROLS: usize = 4; // up/down + rotate left/right

const N_SENSORS: usize = 5;
const SENSOR_SPREAD: f32 = PI / 4.0; // Angle between the first and last sensor

const ENERGY_DRAIN_RATE: f32 = 0.1; // How much energy is lost each tick


pub struct SurvivalTask {
    pub agent: Agent,
    pub food: Vec<Food>,
    pub ticks: u32,

    pub setup: SurvivalTaskSetup
}

#[derive(Copy, Clone)]
pub struct SurvivalTaskSetup {
    pub food_spawn_rate: u32
}

#[derive(Debug)]
pub struct SurvivalTaskResult {
    pub time: u32
}


impl Task for SurvivalTask {
    type Setup = SurvivalTaskSetup;
    type Result = SurvivalTaskResult;

    fn new(setup: &SurvivalTaskSetup) -> SurvivalTask {
        let mut food: Vec<Food> = vec![];

        food.push(Food::new(Agent::START_POS.0 as f32, (Agent::START_POS.1 - 100) as f32));

        food.push(Food::new(100.0,100.0));
        food.push(Food::new(800.0,100.0));
        food.push(Food::new(100.0,800.0));
        food.push(Food::new(800.0,800.0));

        SurvivalTask {
            agent: Agent::new(),
            food,
            ticks: 0,
            setup: *setup
        }
    }
    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState<SurvivalTaskResult> {
        if self.ticks >= Self::MAX_T || self.agent.energy <= 0.0 {
            return self.end_state();
        }

        if self.ticks % self.setup.food_spawn_rate == 0 {
            self.spawn_food();
        }

        self.parse_input(input);
        self.check_collision();

        let sensor_data: Array1<f32> = self.read_sensors();

        self.agent.energy -= ENERGY_DRAIN_RATE;

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

impl SurvivalTask {
    pub const MAX_T: u32 = 2000;

    fn parse_input(&mut self, input: &Vec<TaskInput>) {
        for i in input {
            match i.input_id {
                0 => { self.agent.rotate(0.10); },
                1 => { self.agent.rotate(-0.10); },
                2 => { self.agent.move_faced_direction(Agent::MOVEMENT_SPEED); },
                3 => { self.agent.move_faced_direction(-Agent::MOVEMENT_SPEED); },
                _ => {panic!("invalid input id"); }
            }
        }
    }

    fn check_collision(&mut self) {
        let mut i: usize = 0;

        while i < self.food.len() {
            if self.agent.intersects(&self.food[i]) {
                self.agent.eat(&self.food[i]);

                self.food.remove(i);
            }

            i += 1;
        }
    }

    fn end_state(&self) -> TaskState<SurvivalTaskResult> {
        TaskState {
            result: Some(SurvivalTaskResult {
                time: self.ticks
            }),
            output: TaskOutput { data: Array::zeros(N_SENSORS) }
        }
    }

    fn read_sensors(&self) -> Array1<f32> {
        let mut data: Array1<f32> = Array::zeros(N_SENSORS);

        for f in &self.food {
            for (i, s) in self.agent.sensors.iter().enumerate() {

                let d = s.read((self.agent.x, self.agent.y), (f.x, f.y), Food::RADIUS);
                data[i] = f32::max(data[i], d);
            }
        }

        data
    }

    fn spawn_food(&mut self) {
        let x = random::random_range((100.0, 800.0));
        let y = random::random_range((100.0, 800.0));

        self.food.push(Food::new(x,y));
    }
}

pub struct Agent {
    pub x: f32,
    pub y: f32,
    pub rotation: f32,

    pub energy: f32,

    pub sensors: Vec<Sensor>,
}

impl Agent {
    pub const RADIUS: f32 = 32.0;
    const START_POS: (i16, i16) = (WORLD_SIZE.0 / 2, WORLD_SIZE.1 / 2);
    const START_ROTATION: f32 = PI / 2.0;
    const MOVEMENT_SPEED: f32 = 8.0;

    pub const MAX_ENERGY: f32 = 100.0;

    fn new() -> Agent {
        Agent {
            x: Self::START_POS.0 as f32,
            y: Self::START_POS.1 as f32,
            rotation: Self::START_ROTATION,

            energy: Self::MAX_ENERGY,

            sensors: Self::init_sensors()
        }
    }

    fn eat(&mut self, food: &Food) {
        self.energy += food.energy;
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

    fn intersects(&self, enemy: &Food) -> bool {
        let d = ((self.x - enemy.x).powf(2.0) + (self.y - enemy.y).powf(2.0)).sqrt();

        d < (Self::RADIUS + Food::RADIUS)
    }

    // Move the agent forward/backward in the direction faced
    fn move_faced_direction(&mut self, amount: f32) {
        let dx = self.rotation.cos() * amount;
        let dy = self.rotation.sin() * amount;


        let new_x = self.x + dx;
        let new_y = self.y - dy;

        // TODO: This is imprecise, allowing the agent only to approach the wall but not touch it
        //       Dividing radius by 2 permits stepping into the wall - this avoid that, at least
        if new_x > (WORLD_SIZE.0 as f32) - Self::RADIUS
            || new_x - Self::RADIUS < 0.0
            || new_y > (WORLD_SIZE.1 as f32) - Self::RADIUS
            || new_y - Self::RADIUS < 0.0
        {
            return;
        }

        self.x += dx;
        self.y -= dy;
    }

    fn rotate(&mut self, amount: f32) {
        self.rotation += amount;
        for s in &mut self.sensors {
            s.angle += amount;
        }
    }
}

pub struct Food {
    pub x: f32,
    pub y: f32,

    pub energy: f32
}

impl Food {
    pub const RADIUS: f32 = 8.0;
    pub const BASE_ENERGY: f32 = 20.0;

    fn new(x: f32, y: f32) -> Food {
        Food {
            x,
            y,
            energy: Self::BASE_ENERGY
        }
    }
}

impl TaskEval for SurvivalTask {
    fn eval_setups() -> Vec<SurvivalTaskSetup> {
        vec![SurvivalTaskSetup{ food_spawn_rate: 100 },
             SurvivalTaskSetup{ food_spawn_rate: 200 },
             SurvivalTaskSetup{ food_spawn_rate: 300 }
        ]
    }

    fn fitness(results: Vec<SurvivalTaskResult>) -> f32 {
        let mut f = 0.0;

        for r in &results {
            f += (r.time as f32 / Self::MAX_T as f32) * 100.0;
        }

        f / results.len() as f32
    }
}

