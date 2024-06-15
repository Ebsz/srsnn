use crate::tasks::TaskRenderer;

use tasks::catching_task::{CatchingTask, APPLE_RADIUS, AGENT_RADIUS, ARENA_SIZE};

use sdl2::gfx::primitives::DrawRenderer;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;


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
