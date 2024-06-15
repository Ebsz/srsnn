use crate::tasks::TaskRenderer;

use tasks::survival_task::{SurvivalTask, Agent, Food, WORLD_SIZE, SENSOR_LEN};

use sdl2::gfx::primitives::DrawRenderer;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;
use sdl2::rect::Rect;


impl TaskRenderer for SurvivalTask {
    fn render(&self, canvas: &mut WindowCanvas) {

        // Draw food
        for e in &self.food {
            let _ = canvas.filled_circle(e.x as i16, e.y as i16, Food::RADIUS as i16, Color::RED);
        }

        // Draw Agent
        let _ = canvas.filled_circle(self.agent.x as i16, self.agent.y as i16, Agent::RADIUS as i16, Color::BLACK);

        // Draw Sensors
        for s in &self.agent.sensors {
            let sensor_endpoint = s.endpoint((self.agent.x, self.agent.y));

            let _ = canvas.thick_line(self.agent.x as i16, self.agent.y as i16,
                sensor_endpoint.0 as i16, sensor_endpoint.1 as i16, 2, Color::BLACK);
        }

        let x = self.agent.x + self.agent.rotation.cos() * SENSOR_LEN;
        let y = self.agent.y - self.agent.rotation.sin() * SENSOR_LEN;
        let _ = canvas.thick_line(self.agent.x as i16, self.agent.y as i16, x as i16, y as i16, 3, Color::BLACK);


        let energy_bar_x: i32 = 10;
        let energy_bar_y: i32 = (WORLD_SIZE.1 - 30).into();

        let energy_bar_width: u32 = (800.0 * (self.agent.energy / Agent::MAX_ENERGY)) as u32;
        let energy_bar_height: u32 = 20;

        canvas.set_draw_color(Color::BLUE);
        let _ = canvas.fill_rect(Rect::new(energy_bar_x, energy_bar_y, energy_bar_width, energy_bar_height));
    }

    /// Returns the size of the 'arena' that the task operates in
    fn render_size() -> (i32, i32) {
        (WORLD_SIZE.0 as i32, WORLD_SIZE.1 as i32)
    }
}
