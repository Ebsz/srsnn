//! The classical RL task, where the goal is to keep a pole upright by applying
//! positive and negative force in the x direction
//!
//! End conditions:
//! - Pole angle is less than X dg
//! - Cart moves more than Y units away from the center

use crate::{Task, TaskRenderer, TaskState, TaskInput, TaskOutput, TaskEnvironment, TaskEval};

use utils::{encoding, math};

use sdl2::gfx::primitives::DrawRenderer;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;
use sdl2::rect::Rect;

use ndarray::{array, Array1};

use std::f64::consts::PI;

// Task params
const AGENT_INPUTS: usize = 8;
const AGENT_OUTPUTS: usize = 2;
const WORLD_SIZE: (i32, i32) = (800, 500);


// End conditions
const MAX_ANGLE: f64 = PI / 4.0;
const MAX_T: u32 = 1000;
const MAX_X_CHANGE: f64 = 150.0;

// System parameters
const TC: f64 = 0.003;
const G: f64 = 9.81;

const START_X: f64 = 0.0;

const FORCE: f64 = 100.0;

const POLE_MASS: f64 = 0.1;
const CART_MASS: f64 = 1.0;
const TOTAL_MASS: f64 = POLE_MASS + CART_MASS;
const CT_FRICTION: f64 = 10.01; // Cart-track friction
const CP_FRICTION: f64 = 0.001; // Cart-pole friction

const POLE_START_ANGLE: f64 = -0.05; //PI;
const POLE_LEN: f64 = 100.0;

// Rendering params
const POLE_RENDER_LEN: f64 = 200.0;

const CART_SIZE: (u32, u32) = (100, 30);
const CART_START_POS: (i32, i32) = (WORLD_SIZE.0 / 2 - CART_SIZE.0 as i32/ 2, WORLD_SIZE.1 - 80);


pub struct PoleBalancingSetup { }
pub struct PoleBalancingResult {
    t: u32
}

pub struct PoleBalancingTask {
    cart: Cart,
    pole: Pole,
    t: u32,
}

impl Task for PoleBalancingTask {
    type Setup = PoleBalancingSetup;
    type Result = PoleBalancingResult;


    fn new(setup: &Self::Setup) -> PoleBalancingTask {
        PoleBalancingTask {
            cart: Cart::new(),
            pole: Pole::new(),
            t: 0
        }
    }

    fn tick(&mut self, input: &Vec<TaskInput>) -> TaskState<Self::Result> {
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

    fn get_applied_force(&mut self, input: &Vec<TaskInput>) -> f64 {
        let mut f = 0.0;
        for i in input {
            match i.input_id {
                0 => f -= FORCE,
                1 => f += FORCE,
                _ => {}
            }
        }

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

        let data = array![pos, pos, pos_dot, pos_dot, theta, theta, theta_dot, theta_dot];
        let enc = encoding::rate_encode(data);

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
            fitness += r.t as f32 / MAX_T as f32 * 100.0;
        }

        fitness
    }
}


struct Cart {
    x: f64,
    x_vel: f64,
}

impl Cart {
    fn new() -> Cart {
        Cart {
            x: START_X,
            x_vel: 0.0
        }
    }
}

struct Pole {
    angle: f64,
    angle_vel: f64,
}

impl Pole {
    fn new() -> Pole {
        Pole {
            angle: POLE_START_ANGLE,
            angle_vel: 0.0,
        }
    }
}

impl TaskRenderer for PoleBalancingTask {
    fn render(&self, canvas: &mut WindowCanvas) {
        let _ = canvas.thick_line(0, CART_START_POS.1 as i16 + CART_SIZE.1 as i16 / 2,
            WORLD_SIZE.0 as i16, CART_START_POS.1 as i16 + CART_SIZE.1 as i16 / 2,
            1, Color::RGB(200,200,200));

        // Cart
        let cart_x: i32 = CART_START_POS.0 + self.cart.x.round() as i32;

        let _ = canvas.set_draw_color(Color::BLACK);
        let _ = canvas.fill_rect(Rect::new(cart_x ,CART_START_POS.1, CART_SIZE.0,CART_SIZE.1));

        let angle = self.pole.angle - PI / 2.0;

        // Pole
        let pole_x0: i16 = cart_x as i16 + CART_SIZE.0 as i16 /2;
        let pole_y0: i16 = CART_START_POS.1 as i16;

        let pole_x1: i16 = pole_x0 + (angle.cos() * POLE_RENDER_LEN) as i16;
        let pole_y1: i16 = pole_y0 + (angle.sin() * POLE_RENDER_LEN) as i16;

        let _ = canvas.thick_line(pole_x0, pole_y0, pole_x1, pole_y1, 3, Color::RGB(100,100,100));

        // Pole weight
        let _ = canvas.filled_circle(pole_x1, pole_y1, 12, Color::BLACK);

        // Cart hinge
        let _ = canvas.filled_circle(cart_x as i16 + CART_SIZE.0 as i16 / 2, CART_START_POS.1 as i16, 8, Color::BLACK);
        let _ = canvas.aa_circle(cart_x as i16 + CART_SIZE.0 as i16 / 2, CART_START_POS.1 as i16, 8, Color::BLACK);
    }

    fn render_size() -> (i32, i32) {
       WORLD_SIZE
    }
}
