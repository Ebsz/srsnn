//! The classic RL task, where the goal is to keep a pole upright by applying
//! positive and negative force in the x direction
//!
//! End conditions:
//! - Pole angle is less than X dg
//! - Cart moves more than Y units away from the center

use crate::{Task, TaskState, TaskInput, TaskOutput, TaskEnvironment, TaskEval};

use utils::{encoding, math};

use ndarray::{array, Array1};

use std::f64::consts::PI;

// Task params
const AGENT_INPUTS: usize = 16;
const AGENT_OUTPUTS: usize = 4;

// End conditions
const MAX_ANGLE: f64 = PI / 4.0;
const MAX_T: u32 = 5000;
const MAX_X_CHANGE: f64 = 150.0;

// System parameters
const TC: f64 = 0.005;
const G: f64 = 9.81;

const START_X: f64 = 0.0;
const START_ANGLE: f64 = -0.05;

const FORCE: f64 = 10.0;

const POLE_MASS: f64 = 0.1;
const CART_MASS: f64 = 1.0;
const TOTAL_MASS: f64 = POLE_MASS + CART_MASS;
const CT_FRICTION: f64 = 0.01; // Cart-track friction
const CP_FRICTION: f64 = 0.001; // Cart-pole friction

const POLE_LEN: f64 = 100.0;


#[derive(Clone)]
pub struct PoleBalancingSetup { }

#[derive(Debug)]
pub struct PoleBalancingResult {
    pub t: u32
}

pub struct PoleBalancingTask {
    pub cart: Cart,
    pub pole: Pole,
    pub t: u32,
}

impl Task for PoleBalancingTask {
    type Setup = PoleBalancingSetup;
    type Result = PoleBalancingResult;

    fn new(_setup: &Self::Setup) -> PoleBalancingTask {
        PoleBalancingTask {
            cart: Cart::new(),
            pole: Pole::new(),
            t: 0
        }
    }

    fn tick(&mut self, input: TaskInput) -> TaskState<Self::Result> {
        let force: f64 = self.get_applied_force(input);
        self.update_state(force);

        self.t += 1;

        let mut result = None;

        //println!("{:?}, {:?}, {:?}, {:?}", self.cart.x, self.cart.x_vel, self.pole.angle, self.pole.angle_vel);

        if self.pole.angle.abs() > MAX_ANGLE || self.t >= MAX_T
            || self.cart.x.abs() >= MAX_X_CHANGE {
            result = Some(PoleBalancingResult {
                t: self.t
            });
        }

        TaskState {
            result,
            output: TaskOutput {
                data: self.get_output()
            }
        }
    }

    fn reset(&mut self) {
        self.t = 0;
        self.cart = Cart::new();
        self.pole = Pole::new();
    }

    fn environment() -> TaskEnvironment {
        TaskEnvironment {
            agent_inputs: AGENT_INPUTS,
            agent_outputs: AGENT_OUTPUTS,
        }
    }
}

impl PoleBalancingTask {
    /// With recommended equations
    fn update_state(&mut self, force: f64) {
        let m = POLE_MASS;

        let angle = self.pole.angle;
        let l_hat = POLE_LEN / 2.0;

        let dx = self.cart.x_vel;       // first derivative of x
        let da = self.pole.angle_vel;   // 1st derivative of angle (theta)

        // 1. update x, theta
        self.cart.x += self.cart.x_vel;
        self.pole.angle += self.pole.angle_vel;

        // 2. Eval ddx, dda
        let ddx = (m * G * angle.sin() * angle.cos() - (7.0 / 3.0) * (force + m * l_hat * dx.powf(2.0) * angle.sin() - CT_FRICTION * dx) - ((CP_FRICTION * da * angle.cos()) / l_hat))
        / (m * angle.cos().powf(2.0) - (7.0 / 3.0) * TOTAL_MASS);
        let dda = 3.0 / (7.0 * l_hat) * (G * angle.sin() - ddx * angle.cos() - (CP_FRICTION * da)/(m * l_hat));

        // 3. Update dx, da from ddx, dda
        self.cart.x_vel += ddx * TC;
        self.pole.angle_vel += dda * TC;
    }

    fn get_applied_force(&mut self, input: TaskInput) -> f64 {
        let mut f = 0.0;

        // For 4 agent outputs
        for i in input.data {
            match i {
                0 | 1 => f -= FORCE,
                2 | 3 => f += FORCE,
                _ => { panic!("pole_balancing task got unexpected input {i}"); }
            }
        }

        //let n_per_output = (AGENT_OUTPUTS / 2) as u32;

        //for i in input.data {
        //    println!("input: {i}");
        //    println!("n_per: {n_per_output}");
        //    if i < n_per_output as u32 {
        //        f += 1.0;
        //        println!("plus");

        //    } else if i > n_per_output  && i < n_per_output*2 {
        //        f -= 1.0;
        //        println!("minus");
        //    } else  {
        //        panic!("pole balancing task got unexpected input: {i}");
        //    }
        //}

        f
    }

    fn get_output(&self) -> Array1<f32> {
        // These are approximations found by experimentation, subject to change
        const MAX_THETA_DOT: f64 = 0.025;
        const MAX_POS_DOT: f64 = 5.0;

        let pos: f32 = math::clamp(normalize(self.cart.x, -MAX_X_CHANGE, MAX_X_CHANGE), 0.0, 1.0) as f32;
        let pos_dot: f32 = math::clamp(normalize(self.cart.x_vel, -MAX_POS_DOT, MAX_POS_DOT), 0.0, 1.0) as f32;

        let theta: f32 = math::clamp(normalize(self.pole.angle, -MAX_ANGLE, MAX_ANGLE), 0.0, 1.0) as f32;
        let theta_dot: f32 = math::clamp(normalize(self.pole.angle_vel, -MAX_THETA_DOT, MAX_THETA_DOT), 0.0, 1.0) as f32;

        let data = array![pos, pos, pos, pos,
        pos_dot, pos_dot, pos_dot, pos_dot,
        theta, theta, theta, theta,
        theta_dot, theta_dot, theta_dot, theta_dot];
        let enc = encoding::rate_encode(&data);

        //println!("{:.2}, {:.2}, {:.2}, {:.2}", pos, pos_dot, theta, theta_dot);
        //println!("{:.2}, {:.2}, {:.2}, {:.2}", enc[0], enc[1], enc[2], enc[3]);

        enc
    }
}

fn normalize(x: f64, min: f64, max: f64) -> f64 {
    (x - min) / (max - min)
}

impl TaskEval for PoleBalancingTask {
    fn eval_setups() -> Vec<PoleBalancingSetup> {
        vec![PoleBalancingSetup {}]
    }

    fn fitness(results: Vec<PoleBalancingResult>) -> f32 {
        let mut fitness = 0.0;

        for r in results {
            fitness += r.t as f32; // MAX_T as f32 * 100.0;
        }

        fitness
    }

    fn accuracy(results: &[Self::Result]) -> Option<f32> {
        None
    }
}

pub struct Cart {
    pub x: f64,
    pub x_vel: f64,
}

impl Cart {
    fn new() -> Cart {
        Cart {
            x: START_X,
            x_vel: 0.0
        }
    }
}

pub struct Pole {
    pub angle: f64,
    pub angle_vel: f64,
}

impl Pole {
    fn new() -> Pole {
        Pole {
            angle: START_ANGLE,
            angle_vel: 0.0,
        }
    }
}
