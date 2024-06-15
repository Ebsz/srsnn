use crate::tasks::TaskRenderer;

use tasks::pole_balancing_task::PoleBalancingTask;

use sdl2::gfx::primitives::DrawRenderer;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;
use sdl2::rect::Rect;

use std::f64::consts::PI;

const CART_START_POS: (i32, i32) = (WORLD_SIZE.0 / 2 - CART_SIZE.0 as i32/ 2, WORLD_SIZE.1 - 80);
const CART_SIZE: (u32, u32) = (100, 30);

const WORLD_SIZE: (i32, i32) = (800, 500);
const POLE_RENDER_LEN: f64 = 200.0;


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
